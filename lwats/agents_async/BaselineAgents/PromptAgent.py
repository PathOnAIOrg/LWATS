from typing import Dict
from openai import OpenAI
from ...webagent_utils_async.action.highlevel import HighLevelActionSet
from ...webagent_utils_async.action.utils import execute_action
from ...webagent_utils_async.action.prompt_functions import extract_top_actions, is_goal_finished
from ...webagent_utils_async.browser_env.observation import extract_page_info
from ...webagent_utils_async.evaluation.feedback import capture_post_action_feedback
import time
import logging
import os
import json

logger = logging.getLogger(__name__)

openai_client = OpenAI()


class PromptAgent:

    async def send_prompt(self, plan: str) -> Dict:
        if plan is not None:
            self.messages.append({"role": "user", "content": "The plan is: {}".format(plan)})
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
            model="gpt-4o",
            messages=messages,
        )
        summary = response.choices[0].message.content
        return trajectory,summary

    def __init__(self, messages, goal, images, playwright_manager, features, elements_filter, branching_factor, log_folder,
                 default_model="gpt-4o-mini",
                 planning_model="gpt-4o",
                 action_generation_model="gpt-4o-mini",
                 action_grounding_model="gpt-4o",
                 evaluation_model="gpt-4o"):
        self.action_generation_model = action_generation_model
        self.action_grounding_model = action_grounding_model
        self.evaluation_model = evaluation_model
        self.planning_model = planning_model
        self.default_model = default_model
        self.messages = messages
        self.goal = goal
        self.images = images
        self.playwright_manager = playwright_manager
        self.features = features
        self.elements_filter = elements_filter
        self.branching_factor = branching_factor
        self.messages.append({"role": "user", "content": "The goal is:{}".format(self.goal)})
        self.agent_type = ["bid", "nav", "file", "select_option"]
        self.action_set = HighLevelActionSet(
            subsets=self.agent_type,
            strict=False,
            multiaction=True,
            demo_mode="default"
        )
        self.log_folder = log_folder
        

    async def send_completion_request(self, plan: str, depth: int = 0, trajectory=[]) -> Dict:
        # Increase depth limit to allow more steps (like clicking Buy Now)
        if depth >= 10:
            # Save a default score of 0 if we reached the depth limit without completing
            self._save_default_score(trajectory)
            return trajectory

        context = await self.playwright_manager.get_context()
        page = await self.playwright_manager.get_page()
        pre_action_url = page.url
        # Extract page information
        time.sleep(3)
        page_info =  await extract_page_info(page, fullpage=True, log_folder=self.log_folder)
        updated_actions = await extract_top_actions(trajectory, self.goal, self.images, page_info, self.action_set, openai_client,
                                              self.features, self.elements_filter, self.branching_factor, self.log_folder, fullpage=True, action_generation_model=self.action_generation_model, action_grounding_model=self.action_grounding_model)
        next_action = updated_actions[0]['action']
        await execute_action(next_action, self.action_set, page, context, self.goal, page_info['interactive_elements'],
                       self.log_folder)
        feedback = await capture_post_action_feedback(page, next_action, self.goal, self.log_folder)
        ## TODO 1: save actions!!!
        ## TODO 2: add url, before and after action
        #action_logger = FileLogger(base_folder=os.path.join(self.log_folder, "actions"), prefix="action")
        context = await self.playwright_manager.get_context()
        page = await self.playwright_manager.get_page()
        # action_logger.log(f"Context Info: {context}", "Context")
        # action_logger.log(f"Page Info: {page}", "Page")
        # action_logger.log("Model Responses", f"Raw Responses:\n{updated_actions}")
        # action_logger.log("Model Responses", f"Raw Responses:\n{updated_actions}")
        # action_logger.log(f"Action: {updated_actions[0]}", "Action")
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
        
        # Check for WebShop completion with "Thank you for shopping with us!" and score on page
        webshop_completed = False
        webshop_score = None
        try:
            # Check if this is a WebShop task by checking the URL or content
            content = await page.content()
            if "fixed_" in page.url or ("webshop" in page.url.lower()) or ("Thank you for shopping with us!" in content):
                # Try to detect completion markers
                thank_you_locator = page.locator("text=Thank you for shopping with us!")
                score_locator = page.locator("#reward")
                
                thank_you_count = await thank_you_locator.count()
                score_count = await score_locator.count()
                
                if thank_you_count > 0 and score_count > 0:
                    webshop_completed = True
                    score_text = await score_locator.text_content()
                    webshop_score = score_text.strip()
                    logger.info(f"WebShop completion detected with score: {webshop_score}")
                    
                    # Save score to result file
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
            # If WebShop completion detected, add score to final result
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
        """Save a default score of 0 if the task wasn't completed properly"""
        try:
            # Only save default score for WebShop tasks
            post_action_url = trajectory[-1]['post_action_url'] if trajectory else ""
            if "fixed_" in post_action_url or "webshop" in post_action_url.lower():
                result_file = os.path.join(self.log_folder, 'webshop_score.json')
                
                # Check if score already exists
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
                
                # Add completion message to trajectory
                trajectory.append({
                    'action': 'TERMINATE', 
                    'action_description': 'WebShop task timed out', 
                    'action_result': "Task incomplete. Default score: 0.0",
                    "pre_action_url": post_action_url, 
                    "post_action_url": post_action_url
                })
        except Exception as e:
            logger.error(f"Error saving default WebShop score: {str(e)}")
