import argparse
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import json
import datetime
import logging
from playwright.async_api import async_playwright

from lwats.core_async.config import AgentConfig, add_agent_config_arguments, filter_valid_config_args
from lwats.core_async.agent_factory import setup_search_agent, setup_prompting_web_agent
from lwats.webagent_utils_async.utils.playwright_manager import setup_playwright
from lwats.replay_async import playwright_step_execution

# Simplified instruction extraction function
async def extract_instructions_from_webpage(url, browser_mode="chromium"):
    """
    Extract task instructions from the WebShop page with simplified logic.
    
    Args:
        url (str): URL of the WebShop task page
        browser_mode (str): Browser engine to use (chromium/browserbase)
        
    Returns:
        str: Extracted instructions text
    """
    # Ensure browser_mode is never None
    if browser_mode is None:
        browser_mode = "chromium"
        
    async with async_playwright() as p:
        # Choose browser engine based on browser_mode parameter
        if browser_mode.lower() == "browserbase":
            browser = await p.chromium.launch(headless=True)
        else:  # Default to chromium
            browser = await p.chromium.launch(headless=True)
            
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle")
            
            # Try common instruction selectors
            possible_selectors = [
                "div.instruction-text", "div.instruction", "div.task-instruction",
                "div:has-text('Instruction:')", "div:has-text('Task:')"
            ]
            
            instruction_text = None
            for selector in possible_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=1000)
                    if element:
                        instruction_text = await element.inner_text()
                        if instruction_text and len(instruction_text.strip()) > 10:
                            instruction_text = instruction_text.strip()
                            # Clean up text
                            for prefix in ["WebShop\nInstruction:", "Instruction:", "Task:"]:
                                if instruction_text.startswith(prefix):
                                    instruction_text = instruction_text[len(prefix):].strip()
                                    break
                            return instruction_text
                except Exception:
                    continue
            
            # If we couldn't find instructions, return a generic message
            return "Explore the WebShop interface and complete the shopping task based on the product requirements shown on the page."
        finally:
            await browser.close()

def get_webshop_score(log_folder):
    """
    Get WebShop score from the webshop_score.json file created by PromptAgent.
    
    Args:
        log_folder (str): Path to the log folder
        
    Returns:
        float: The score (0.0 if no score file exists or score can't be parsed)
    """
    score_file = os.path.join(log_folder, 'webshop_score.json')
    if not os.path.exists(score_file):
        return 0.0
        
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
            
        # Extract numeric score from score text (format like "Your score (min 0.0, max 1.0): 0.75")
        score_text = score_data.get('score', '0.0')
        import re
        score_match = re.search(r'(\d+\.\d+)', score_text)
        if score_match:
            return float(score_match.group(1))
    except Exception as e:
        logging.error(f"Error reading WebShop score: {str(e)}")
    
    return 0.0

def setup_logger(task_id, log_folder="log"):
    """Set up logging for a specific task with both file and console handlers."""
    logger = logging.getLogger(f"{task_id}")
    logger.setLevel(logging.INFO)
    os.makedirs(log_folder, exist_ok=True)
    log_fh = logging.FileHandler(os.path.join(log_folder, f'{task_id}.log'), encoding='utf-8')
    log_fh.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(message)s')
    terminal_format = logging.Formatter('%(message)s')
    log_fh.setFormatter(log_format)
    console_handler.setFormatter(terminal_format)
    logger.addHandler(log_fh)
    logger.addHandler(console_handler)
    return logger, log_fh, console_handler

async def run_tree_search(headless, browser_mode, starting_url, agent_type, goal, 
                    search_algorithm, action_generation_model, images, task_id, agent_config):
    """
    First phase: Run tree search to find the best trajectory.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        browser_mode (str): Browser mode (chromium/browserbase)
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use
        goal (str): Task goal/instruction
        search_algorithm (str): Search algorithm to use (bfs/dfs)
        action_generation_model (str): Model to use for action generation
        images (list): List of image paths
        task_id (str): Task ID for logging
        agent_config (AgentConfig): Agent configuration
        
    Returns:
        dict: Search results containing the best trajectory
    """
    log_folder = os.path.join("log", task_id)
    logger = logging.getLogger(f"{task_id}")
    logger.info(f"Starting tree search with algorithm: {search_algorithm}")
    logger.info(f"Using browser mode: {browser_mode}")
    logger.info(f"Max depth: {agent_config.max_depth}")
    
    # Configure search algorithm and browser
    agent_config.search_algorithm = search_algorithm
    agent_config.browser_mode = browser_mode
    agent_config.headless = headless
    
    # Ensure we're using SimpleSearchAgent
    if agent_type != "SimpleSearchAgent":
        logger.warning(f"Agent type {agent_type} is not supported for this script. Using SimpleSearchAgent instead.")
        agent_type = "SimpleSearchAgent"
    
    # Set up search agent
    logger.info(f"Setting up search agent of type: {agent_type} with {search_algorithm} algorithm")
    search_agent_result = await setup_search_agent(
        agent_type=agent_type,
        starting_url=starting_url,
        goal=goal,
        images=images,
        agent_config=agent_config
    )
    
    # Handle the return value based on whether it's a tuple or just an agent
    if isinstance(search_agent_result, tuple) and len(search_agent_result) == 2:
        agent, playwright_manager = search_agent_result
    else:
        # If only one value was returned, it's the agent
        agent = search_agent_result
        playwright_manager = getattr(agent, 'playwright_manager', None)
    
    try:
        # Run the search
        logger.info(f"Running search with agent of type: {type(agent).__name__}")
        search_results = await agent.run()
        
        # Handle different return types from search agents
        if hasattr(search_results, 'best_path') and hasattr(search_results, 'score'):
            # This is a LATSNode or similar object with a best_path attribute
            best_trajectory = search_results.best_path
            search_score = getattr(search_results, 'score', 0.0)
            logger.info(f"Tree search completed. Found trajectory with {len(best_trajectory)} steps and score {search_score}")
            
            # Convert to a dictionary format for consistency
            results_dict = {
                'best_trajectory': best_trajectory,
                'score': search_score,
                'search_algorithm': search_algorithm
            }
        elif hasattr(search_results, 'get_trajectory') and hasattr(search_results, 'value'):
            # This is a LATSNode object from SimpleSearchAgent
            best_trajectory = search_results.get_trajectory()
            search_score = getattr(search_results, 'value', 0.0)
            logger.info(f"Tree search completed. Found trajectory with {len(best_trajectory)} steps and score {search_score}")
            
            # Convert to a dictionary format for consistency
            results_dict = {
                'best_trajectory': best_trajectory,
                'score': search_score,
                'search_algorithm': search_algorithm
            }
        elif isinstance(search_results, dict) and 'best_trajectory' in search_results:
            # This is a dictionary with best_trajectory key
            best_trajectory = search_results.get('best_trajectory', [])
            search_score = search_results.get('score', 0.0)
            logger.info(f"Tree search completed. Found trajectory with {len(best_trajectory)} steps")
            results_dict = search_results
        else:
            # Fallback: create a default dictionary if the structure is unknown
            logger.warning(f"Unknown search results structure. Creating default dictionary.")
            results_dict = {
                'best_trajectory': [],
                'score': 0.0,
                'search_algorithm': search_algorithm
            }
        
        # Save search results
        results_file = os.path.join(log_folder, 'search_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4)
        
        return results_dict
    finally:
        # Close the playwright_manager when done, if it exists
        if playwright_manager:
            await playwright_manager.close()
        elif hasattr(agent, 'close') and callable(agent.close):
            await agent.close()

async def execute_trajectory(headless, browser_mode, starting_url, agent_type, goal, 
                      action_generation_model, images, trajectory, task_id):
    """
    Second phase: Execute the best trajectory found by tree search.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        browser_mode (str): Browser mode (chromium/browserbase)
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use
        goal (str): Task goal/instruction
        action_generation_model (str): Model to use for action generation
        images (list): List of image paths
        trajectory (list): Best trajectory from tree search
        task_id (str): Task ID for logging
        
    Returns:
        tuple: (trajectory, result, score)
    """
    log_folder = os.path.join("log", task_id)
    logger = logging.getLogger(f"{task_id}")
    logger.info(f"Executing best trajectory with {len(trajectory)} steps")
    
    # Setup playwright
    playwright_manager = await setup_playwright(
        headless=headless,
        mode=browser_mode,
        storage_state=None  # No storage state for WebShop
    )
    
    try:
        page = await playwright_manager.get_page()
        await page.goto(starting_url)
        
        # Execute each step in the trajectory
        exec_result = []
        for i, step in enumerate(trajectory):
            logger.info(f"Executing step {i+1}/{len(trajectory)}")
            
            # Extract action from step
            if isinstance(step, dict):
                action = step.get("action")
                description = step.get("natural_language_description", "")
            else:
                action = step
                description = ""
            
            logger.info(f"Action: {action}, Description: {description}")
            
            # Record pre-action URL
            pre_action_url = page.url
            
            # Execute the action using playwright_step_execution
            try:
                # Create a simplified node object with the minimum required attributes
                node = type('Node', (), {
                    'action': action,
                    'natural_language_description': description,
                    'element': None,  # Will be located by playwright_step_execution
                })
                
                success = await playwright_step_execution(
                    node,
                    goal,
                    playwright_manager,
                    is_replay=False,
                    log_folder=log_folder
                )
                
                # Capture result
                post_action_url = page.url
                step_result = {
                    'action': action,
                    'action_description': description,
                    'action_result': 'Success' if success else 'Failed',
                    'pre_action_url': pre_action_url,
                    'post_action_url': post_action_url
                }
                exec_result.append(step_result)
                
                # Wait a bit for page to stabilize
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error executing action {action}: {str(e)}")
                step_result = {
                    'action': action,
                    'action_description': description,
                    'action_result': f"Error: {str(e)}",
                    'pre_action_url': pre_action_url,
                    'post_action_url': page.url
                }
                exec_result.append(step_result)
        
        # Check for WebShop completion and get score
        webshop_score = None
        try:
            content = await page.content()
            if "fixed_" in page.url or ("webshop" in page.url.lower()) or ("Thank you for shopping with us!" in content):
                thank_you_locator = page.locator("text=Thank you for shopping with us!")
                score_locator = page.locator("#reward")
                
                thank_you_count = await thank_you_locator.count()
                score_count = await score_locator.count()
                
                if thank_you_count > 0 and score_count > 0:
                    score_text = await score_locator.text_content()
                    webshop_score = score_text.strip()
                    logger.info(f"WebShop completion detected with score: {webshop_score}")
                    
                    # Save score to result file
                    try:
                        result_file = os.path.join(log_folder, 'webshop_score.json')
                        score_data = {
                            "score": webshop_score,
                            "url": page.url,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(score_data, f, indent=4)
                        logger.info(f"WebShop score saved to {result_file}")
                    except Exception as e:
                        logger.error(f"Error saving WebShop score: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking WebShop completion: {str(e)}")
        
        # Get the score
        score = get_webshop_score(log_folder)
        logger.info(f"Final score: {score}")
        
        return trajectory, exec_result, score
    finally:
        # Close the playwright_manager when done
        await playwright_manager.close()

async def main(headless=False, browser_mode="chromium", starting_url=None, agent_type="SimpleSearchAgent", goal=None, 
         search_algorithm="bfs", action_generation_model="gpt-4o", images=None, task_id=None, max_depth=2, **kwargs):
    """
    Main function to run WebShop tree search and evaluation.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        browser_mode (str): Browser mode (chromium/browserbase)
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use (only SimpleSearchAgent is supported)
        goal (str): Task goal/instruction (optional, will be extracted from page if None)
        search_algorithm (str): Algorithm to use for search (bfs/dfs)
        action_generation_model (str): Model to use for action generation
        images (list): List of image paths
        task_id (str): Optional task ID for logging
        max_depth (int): Maximum depth for tree search (default: 2)
        **kwargs: Additional arguments for agent configuration
    """
    # Ensure default values for parameters that might be None
    if starting_url is None:
        starting_url = "http://128.105.144.173:3000/fixed_0"
    if images is None:
        images = []
    if browser_mode is None:
        browser_mode = "chromium"
    if max_depth is None:
        max_depth = 2

    # Setup logging
    if task_id:
        log_folder = os.path.join("log", task_id)
        os.makedirs(log_folder, exist_ok=True)
        logger, log_fh, console_handler = setup_logger(task_id, log_folder)
        logger.info(f"Starting WebShop tree search evaluation for task {task_id}")
        logger.info(f"Starting URL: {starting_url}")
        logger.info(f"Search algorithm: {search_algorithm}")
        logger.info(f"Browser mode: {browser_mode}")
        logger.info(f"Max depth: {max_depth}")
    else:
        log_folder = "log"
        logger = logging.getLogger("default")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        log_fh = None

    # If goal is None or empty, extract it from the webpage
    if not goal:
        logger.info("Extracting instructions from webpage...")
        extracted_goal = await extract_instructions_from_webpage(starting_url, browser_mode)
        goal = extracted_goal
        logger.info(f"Extracted goal: {goal[:100]}...")  # Log first 100 chars for verification
    
    # Create agent configuration
    config_dict = filter_valid_config_args(kwargs)
    agent_config = AgentConfig(**config_dict)
    # Explicitly set max_depth to ensure it's not None
    agent_config.max_depth = max_depth
    
    try:
        # Phase 1: Run tree search to find best trajectory
        search_results = await run_tree_search(
            headless=headless,
            browser_mode=browser_mode,
            starting_url=starting_url,
            agent_type=agent_type,
            goal=goal,
            search_algorithm=search_algorithm,
            action_generation_model=action_generation_model,
            images=images,
            task_id=task_id,
            agent_config=agent_config
        )
        
        # Extract best trajectory from the results_dict returned by run_tree_search
        best_trajectory = search_results.get('best_trajectory', [])
        if not best_trajectory:
            logger.error("No valid trajectory found by tree search")
            return None, "No valid trajectory found", 0.0
            
        logger.info(f"Using best trajectory with {len(best_trajectory)} steps")
        for i, step in enumerate(best_trajectory):
            if isinstance(step, dict):
                logger.info(f"Step {i+1}: {step.get('action', '?')} - {step.get('natural_language_description', '?')}")
            else:
                logger.info(f"Step {i+1}: {step}")
        
        # Phase 2: Execute the best trajectory
        trajectory, result, score = await execute_trajectory(
            headless=headless,
            browser_mode=browser_mode,
            starting_url=starting_url,
            agent_type=agent_type,
            goal=goal,
            action_generation_model=action_generation_model,
            images=images,
            trajectory=best_trajectory,
            task_id=task_id
        )
        
        # Save final results
        if task_id:
            result_file = os.path.join(log_folder, 'result.json')
            final_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "trajectory": trajectory,
                "result": result,
                "score": score,
                "search_algorithm": search_algorithm,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type,
                "action_generation_model": action_generation_model,
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=4)
        
        return trajectory, result, score

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if task_id:
            result_file = os.path.join(log_folder, 'error.json')
            error_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "error": str(e),
                "score": 0.0,  # Default score of 0 for errors
                "search_algorithm": search_algorithm,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type,
                "action_generation_model": action_generation_model
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_json, f, indent=4)
        raise
    finally:
        if log_fh:
            logger.removeHandler(log_fh)
        logger.removeHandler(console_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebShop tree search evaluation")
    
    # Add agent config arguments first to know which arguments are already defined
    add_agent_config_arguments(parser)
    
    # Now add our custom arguments, avoiding any duplicates from add_agent_config_arguments
    parser.add_argument("--headless", type=bool, default=False,
                        help="Specify if the browser should run in headless mode (default: False)")
    # If browser-mode is not added by add_agent_config_arguments, add it here
    if not any(action.dest == 'browser_mode' for action in parser._actions):
        parser.add_argument("--browser-mode", type=str, default="chromium", choices=["chromium", "browserbase"],
                            help="Specify the browser mode: chromium or browserbase (default: chromium)")
    parser.add_argument("--starting-url", type=str, default="http://128.105.144.173:3000/fixed_0",
                        help="Starting URL for the web agent (default: http://128.105.144.173:3000/fixed_0)")
    parser.add_argument("--agent-type", type=str, default="SimpleSearchAgent",
                        help="Type of agent to use (default: SimpleSearchAgent)")
    parser.add_argument("--goal", type=str, default=None,
                        help="Goal for the web agent to accomplish (if not provided, will be extracted from webpage)")
    parser.add_argument("--search-algorithm", type=str, default="bfs", choices=["bfs", "dfs"],
                        help="Search algorithm to use (default: bfs)")
    parser.add_argument("--action-generation-model", type=str, default="gpt-4o",
                        help="Action generation model (default: gpt-4o)")
    parser.add_argument("--images", type=str, default="",
                        help="Comma-separated paths to image files (e.g., 'path1.jpg,path2.jpg')")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Task ID for this evaluation run")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Maximum depth for tree search (default: 2)")
    parser.add_argument("--batch-start", type=int, default=None,
                        help="Start task number for batch evaluation")
    parser.add_argument("--batch-end", type=int, default=None,
                        help="End task number for batch evaluation")
    
    args = parser.parse_args()
    
    # Convert images string to list
    images_list = [img.strip() for img in args.images.split(',')] if args.images else []
    
    # Ensure browser_mode is set
    if not hasattr(args, 'browser_mode') or args.browser_mode is None:
        args.browser_mode = "chromium"
    
    # Handle batch evaluation
    if args.batch_start is not None and args.batch_end is not None:
        base_url = args.starting_url.rstrip('/')
        # Get the base URL before the task number
        if '_' in base_url:
            base_url = base_url.rsplit('_', 1)[0]
        
        for task_num in range(args.batch_start, args.batch_end + 1):
            task_url = f"{base_url}_{task_num}"
            task_id = f"webshop_simple_search_{args.search_algorithm}_{task_num}"
            print(f"\nRunning WebShop search task {task_num} at {task_url} with {args.search_algorithm}")
            try:
                # Create a modified copy of args.__dict__ with the task-specific values
                task_args = args.__dict__.copy()
                task_args['starting_url'] = task_url
                task_args['task_id'] = task_id
                task_args['images'] = images_list
                
                # Call main with task-specific kwargs only
                trajectory, result, score = asyncio.run(main(**task_args))
                print(f"Completed task {task_num} with score {score}")
            except Exception as e:
                print(f"Error in task {task_num}: {str(e)}")
                continue
    else:
        # Single task evaluation
        task_id = args.task_id or f"webshop_simple_search_{args.search_algorithm}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create kwargs from args.__dict__ and update with task_id and images_list
        task_args = args.__dict__.copy()
        task_args['task_id'] = task_id
        task_args['images'] = images_list
        
        # Call main with kwargs only
        trajectory, result, score = asyncio.run(main(**task_args))
        print(f"Final score: {score}")