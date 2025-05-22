from typing import Dict
from openai import OpenAI
from ...webagent_utils_async.action.highlevel import HighLevelActionSet
from ...webagent_utils_async.action.utils import execute_action
from ...webagent_utils_async.action.prompt_functions import extract_top_actions, is_goal_finished
from ...webagent_utils_async.browser_env.observation import extract_page_info
from ...webagent_utils_async.evaluation.feedback import capture_post_action_feedback
from ...core_async.config import PromptingAgentConfig
import time
import logging
import os
import json

logger = logging.getLogger(__name__)

openai_client = OpenAI()


class PromptAgent:

    def __init__(self, messages: list, goal: str, images: list, playwright_manager, config: PromptingAgentConfig):
        """
        Initialize PromptAgent with configuration.
        
        Args:
            messages: List of message dictionaries
            goal: The goal for the agent
            images: List of image paths
            playwright_manager: Playwright manager instance
            config: PromptingAgentConfig instance
        """
        # Store config
        self.config = config
        
        # Store model settings
        self.action_generation_model = config.action_generation_model
        self.action_grounding_model = config.action_grounding_model
        self.evaluation_model = config.evaluation_model
        self.planning_model = config.planning_model
        self.default_model = config.default_model
        
        # Store basic settings
        self.messages = messages
        self.goal = goal
        self.images = images
        self.playwright_manager = playwright_manager
        
        # Store feature settings
        self.features = config.features
        self.elements_filter = config.elements_filter
        self.branching_factor = config.branching_factor
        
        # Add goal to messages
        self.messages.append({"role": "user", "content": "The goal is:{}".format(self.goal)})
        
        # Setup action set
        self.agent_type = ["bid", "nav", "file", "select_option"]
        self.action_set = HighLevelActionSet(
            subsets=self.agent_type,
            strict=False,
            multiaction=True,
            demo_mode="default"
        )
        
        # Store logging settings
        self.log_folder = config.log_folder

    async def send_prompt(self, plan: str) -> Dict:
        """
        Send a prompt to the agent and get the response.
        
        Args:
            plan: Optional plan to follow
            
        Returns:
            Tuple of (trajectory, summary)
        """
        if plan is not None:
            self.messages.append({"role": "user", "content": "The plan is: {}".format(plan)})
        else:
            self.messages.append({"role": "user", "content": "The plan is: {}".format(self.goal)})
        trajectory = await self.send_completion_request(plan, 0, [])
        messages = [{"role": "system",
                     "content": "The goal is {}, summarize the actions and result taken by the web agent in one sentence, be concise.".format(
                         self.goal)}]
        for item in trajectory:
            action = item['action']
            action_result = item['action_result']
            messages.append({"role": "user", "content": 'action is: {}'.format(action)})
            messages.append({"role": "user", "content": 'action result is: {}'.format(action_result)})
        response = openai_client.chat.completions.create(
            model=self.default_model,
            messages=messages,
        )
        summary = response.choices[0].message.content
        return trajectory, summary

    async def send_completion_request(self, plan: str, depth: int = 0, trajectory=[]) -> Dict:
        """
        Send a completion request to the agent.
        
        Args:
            plan: Optional plan to follow
            depth: Current depth of the request
            trajectory: Current trajectory of actions
            
        Returns:
            Updated trajectory
        """
        if depth >= 10:
            self._save_default_score(trajectory)
            return trajectory

        context = await self.playwright_manager.get_context()
        page = await self.playwright_manager.get_page()
        pre_action_url = page.url
        
        # Extract page information
        time.sleep(3)
        page_info = await extract_page_info(
            page, 
            fullpage=self.config.fullpage, 
            log_folder=self.log_folder
        )
        
        updated_actions = await extract_top_actions(
            trajectory, 
            self.goal, 
            self.images, 
            page_info, 
            self.action_set, 
            openai_client,
            self.features, 
            self.elements_filter, 
            self.branching_factor, 
            self.log_folder, 
            fullpage=self.config.fullpage, 
            action_generation_model=self.action_generation_model, 
            action_grounding_model=self.action_grounding_model
        )
        
        next_action = updated_actions[0]['action']
        await execute_action(
            next_action, 
            self.action_set, 
            page, 
            context, 
            self.goal, 
            page_info['interactive_elements'],
            self.log_folder
        )
        
        feedback = await capture_post_action_feedback(page, next_action, self.goal, self.log_folder)
        
        post_action_url = page.url

        trajectory.append({
            'action': updated_actions[0]['action'],
            'action_description': updated_actions[0]['natural_language_description'],  
            'action_result': feedback, 
            "pre_action_url": pre_action_url, 
            "post_action_url": post_action_url
        })

        print(f"The action is: {next_action} - The action result is: {feedback}")

        messages = [{"role": "system", "content": "The goal is {}, Is the overall goal finished?".format(self.goal)}]
        for item in trajectory:
            action = item['action']
            action_result = item['action_result']
            messages.append({"role": "user", "content": 'action is: {}'.format(action)})
            messages.append({"role": "user", "content": 'action result is: {}'.format(action_result)})

        goal_finished = await is_goal_finished(messages, openai_client)
        
        # Check for WebShop completion
        webshop_completed = False
        webshop_score = None
        try:
            content = await page.content()
            if "fixed_" in page.url or ("webshop" in page.url.lower()) or ("Thank you for shopping with us!" in content):
                thank_you_locator = page.locator("text=Thank you for shopping with us!")
                score_locator = page.locator("#reward")
                
                thank_you_count = await thank_you_locator.count()
                score_count = await score_locator.count()
                
                if thank_you_count > 0 and score_count > 0:
                    webshop_completed = True
                    score_text = await score_locator.text_content()
                    webshop_score = score_text.strip()
                    logger.info(f"WebShop completion detected with score: {webshop_score}")
                    
                    try:
                        result_file = os.path.join(self.log_folder, 'webshop_score.json')
                        score_data = {
                            "score": webshop_score,
                            "url": page.url,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(score_data, f, indent=4)
                        logger.info(f"WebShop score saved to {result_file}")
                    except Exception as e:
                        logger.error(f"Error saving WebShop score: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking WebShop completion: {str(e)}")

        if webshop_completed or goal_finished:
            if webshop_completed and webshop_score:
                trajectory.append({
                    'action': 'TERMINATE', 
                    'action_description': 'WebShop purchase completed', 
                    'action_result': f"Task complete! {webshop_score}",
                    "pre_action_url": post_action_url, 
                    "post_action_url": post_action_url
                })
            return trajectory

        return await self.send_completion_request(plan, depth + 1, trajectory)
        
    def _save_default_score(self, trajectory):
        """
        Save a default score of 0 if the task wasn't completed properly.
        
        Args:
            trajectory: Current trajectory of actions
        """
        try:
            post_action_url = trajectory[-1]['post_action_url'] if trajectory else ""
            if "fixed_" in post_action_url or "webshop" in post_action_url.lower():
                result_file = os.path.join(self.log_folder, 'webshop_score.json')
                
                if os.path.exists(result_file):
                    return
                
                score_data = {
                    "score": "Your score (min 0.0, max 1.0): 0.0",
                    "url": post_action_url,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "Default score - task not fully completed"
                }
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(score_data, f, indent=4)
                logger.info(f"Default WebShop score (0.0) saved to {result_file}")
                
                trajectory.append({
                    'action': 'TERMINATE', 
                    'action_description': 'WebShop task timed out', 
                    'action_result': "Task incomplete. Default score: 0.0",
                    "pre_action_url": post_action_url, 
                    "post_action_url": post_action_url
                })
        except Exception as e:
            logger.error(f"Error saving default WebShop score: {str(e)}")
