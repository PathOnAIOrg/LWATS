import argparse
from dotenv import load_dotenv
load_dotenv()
from lwats.core_async.agent_factory import setup_search_agent
import asyncio
import os
import json
import datetime
import logging
from lwats.core_async.config import AgentConfig, add_agent_config_arguments, filter_valid_config_args
import webbrowser
import time
from playwright._impl._errors import Error as PlaywrightError
import dataclasses
from typing import List

# Custom JSON encoder to handle non-serializable objects (like LATSNode)
class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return {k: str(v) if isinstance(v, (type, object)) and not 
                                 isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))) 
                                 else v 
                    for k, v in obj.__dict__.items()}
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return super().default(obj)

WEBSHOP_GOAL = """Figure out the task to be done in the first page instruction and then do the task accordingly. 

IMPORTANT SEARCH QUERY GUIDANCE:
1. When searching, use short, general terms (2-4 words) instead of copying the entire product description. For example:
   - GOOD: "black loafers" or "men's shoes" or "rubber sole shoes"
   - BAD: "men's size 10.5 black loafers with rubber soles under $70"

2. Start with broader terms, then narrow down by filtering or browsing specific product pages. Scroll through all results on the first page and identify if there is any product matches the requirement. If there is, click on the product to go to the product page. If there is no product matches the requirement, try to navigate to next page or trya different search query. Search for the main product category first, then check details like size and price on individual product pages.

3. Select the relevant color and size of the product if there is any. Click on "Buy Now" button to go to the checkout page if you are confident that you have found the product that matches the requirement.

4. The page with "Thank you for shopping with us!" and "Your score" is the only page that confirms the task is complete.

BROWSING AND NAVIGATION GUIDANCE:
1. Always scroll down to view all results on the current page. Relevant items may be lower on the page and not immediately visible.

2. If you don't find suitable items on the first page:
   - Look for and click on "Next" button to go to the next page
   - Try at least 2-3 pages of results before changing your search query

3. When examining a product page:
   - Scroll the entire page to see all options, details, and variations
   - Many products have multiple options (sizes, colors) that only appear when browsing the full product page

COMPLETION AND SCORING GUIDANCE:
1. The task is ONLY complete when you see "Your score (min 0.0, max 1.0)" displayed on the page. This confirms your purchase was evaluated.

2. If you've clicked "Buy Now" but don't see "Your score":
   - DO NOT end the task or close the browser
   - You may need to complete additional steps (like confirming purchase)
   - Continue interacting with the page until you see the score

3. If you've explored extensively and can't find a good match:
   - It's better to purchase something that partially matches the requirements than to make no purchase
   - Any purchase (even if imperfect) will score higher than no purchase

Please note that certain options can be chosen inside the product page such as color or size which means the image in the search page is only one example of the product. Also, there might not be a perfect match, in which case you should try to find the closest match as possible. 

The searched result is ranked from the most relevant to the least relevant so usually next page will give less relevant products. If there is no result in the next page, consider going back to search and trying different queries. 

Also, you only have limited number of actions each time, so please use them wisely. If you end up buying nothing, then you will receive zero score. It is better to at least select something that matches imperfectly.

EXAMPLE SHOPPING STRATEGY:
1. Read the full product requirement from the instruction
2. Search with general terms (e.g., "black shoes" for "men's size 10.5 black loafers with rubber soles under $70")
3. Scroll through all results on the first page
4. Check next pages of results if needed
5. Open promising product pages to check details (size, price, material)
6. If no good match is found, try a different search query
7. Select the closest matching product before running out of actions
8. Verify that "Your score" appears after completing the purchase
"""

def setup_logger(task_id, log_folder="log"):
    logger = logging.getLogger(f"{task_id}")
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers if logger already exists
    if not logger.handlers:
        os.makedirs(log_folder, exist_ok=True)
        log_fh = logging.FileHandler(os.path.join(log_folder, f'{task_id}.log'), encoding='utf-8')
        log_fh.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        terminal_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_fh.setFormatter(log_format)
        console_handler.setFormatter(terminal_format)
        logger.addHandler(log_fh)
        logger.addHandler(console_handler)
    return logger # Return the logger instance for direct use

async def main(starting_url, agent_type, goal, images, task_id=None, agent_config_args=None, timeout_minutes=10):
    logger = setup_logger(task_id or "default_task", agent_config_args.log_folder if agent_config_args else "log")
    logger.info(f"Starting evaluation for task {task_id}")
    logger.info(f"Starting URL: {starting_url}")
    logger.info(f"Agent Type: {agent_type}")
    logger.info(f"Setting timeout to {timeout_minutes} minutes")

    playwright_manager = None
    start_time = time.time()
    agent_run_results = {} # Initialize with a default dictionary

    try:
        agent, playwright_manager = await setup_search_agent(
            agent_type=agent_type,
            starting_url=starting_url,
            goal=goal,
            images=images,
            agent_config=agent_config_args # Pass the fully configured AgentConfig object
        )
        
        if agent_config_args.browser_mode == "browserbase":
            await asyncio.sleep(2) 
            try:
                live_browser_url = await playwright_manager.get_live_browser_url()
                session_id = await playwright_manager.get_session_id()
                if live_browser_url:
                    logger.info(f"✅ BrowserBase session created! Session ID: {session_id}")
                    logger.info(f"🔍 View browser actions at: {live_browser_url}")
                    print(f"\n✅ BrowserBase session created! Session ID: {session_id}")
                    print(f"🔍 View browser actions at: {live_browser_url}")
                    webbrowser.open(live_browser_url)
                    await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Error getting BrowserBase URL: {str(e)}", exc_info=True)

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= timeout_minutes:
                    logger.warning(f"Timeout reached ({elapsed_minutes:.1f}min/{timeout_minutes}min). Terminating search.")
                    agent_run_results = {"status": "timeout", "error": f"Search exceeded {timeout_minutes} minute timeout", "webshop_score": "0.0"}
                    break
                
                logger.info(f"Attempting agent.run() (attempt {retry_count + 1}/{max_retries})...")
                remaining_minutes = max(0.1, timeout_minutes - elapsed_minutes)
                logger.info(f"Setting agent run timeout to {remaining_minutes:.1f} minutes")
                
                agent_run_results = await asyncio.wait_for(
                    agent.run(),
                    timeout=remaining_minutes * 60
                )
                # Explicitly check status from agent, agent might determine it's incomplete even if no exception
                if agent_run_results.get("status") == "timeout": # Agent itself might timeout internally
                    logger.warning("Agent reported internal timeout.")
                elif agent_run_results.get("status") == "error":
                     logger.error(f"Agent reported an error: {agent_run_results.get('error')}")
                break # Success or agent-defined terminal state
            except asyncio.TimeoutError:
                logger.warning(f"Task-level timeout for agent.run() after {remaining_minutes:.1f} minutes")
                agent_run_results = {"status": "timeout", "error": "Agent execution timed out at task level", "webshop_score": "0.0"}
                break 
            except PlaywrightError as e:
                logger.error(f"PlaywrightError during agent run: {str(e)}", exc_info=True)
                retry_count += 1
                if "Target page, context or browser has been closed" in str(e) and retry_count < max_retries:
                    logger.warning(f"Browser closed unexpectedly. Retrying ({retry_count}/{max_retries})...")
                    if playwright_manager:
                        try: await playwright_manager.close()
                        except: pass
                    await asyncio.sleep(2)
                    try:
                        agent, playwright_manager = await setup_search_agent(
                            agent_type=agent_type, starting_url=starting_url, goal=goal, images=images, agent_config=agent_config_args
                        )
                        if agent_config_args.browser_mode == "browserbase":
                            await asyncio.sleep(2)
                            try:
                                live_browser_url = await playwright_manager.get_live_browser_url()
                                if live_browser_url: 
                                    print(f"\n✅ New BrowserBase session: {live_browser_url}")
                                    webbrowser.open(live_browser_url)
                                    await asyncio.sleep(3)
                            except Exception as e_url: logger.error(f"Error getting new BB URL: {str(e_url)}")
                    except Exception as setup_err:
                        logger.error(f"Error re-setting up agent: {str(setup_err)}", exc_info=True)
                        agent_run_results = {"status": "error", "error": "Failed to setup agent on retry", "webshop_score": "0.0"}
                        break
                else: 
                    agent_run_results = {"status": "error", "error": f"Unhandled PlaywrightError or max retries: {str(e)}", "webshop_score": "0.0"}
                    break
            except Exception as e_outer:
                logger.error(f"Unexpected error during agent run: {str(e_outer)}", exc_info=True)
                agent_run_results = {"status": "error", "error": f"Unexpected error: {str(e_outer)}", "webshop_score": "0.0"}
                break

        if not agent_run_results: # Should have been set
            agent_run_results = {"status": "unknown_error", "error": "Agent run completed loop without setting results", "webshop_score": "0.0"}
            if retry_count >= max_retries : agent_run_results["error"] = "Max retries reached for browser closed error."

        logger.info("Run Phase Completed. Final Agent Results:")
        try:
            logger.info(json.dumps(agent_run_results, indent=2, cls=NodeEncoder))
        except Exception as e:
            logger.error(f"Error serializing agent run results: {str(e)}", exc_info=True)
            logger.info(f"Raw agent run results (might be unserializable): {str(agent_run_results)}")

        if task_id:
            result_file = os.path.join(agent_config_args.log_folder, task_id, 'result.json')
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # Ensure final_json uses the structured agent_run_results
            final_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "results": agent_run_results, # Use the structured results from agent.run()
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type
            }
            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, indent=4, cls=NodeEncoder)
            except Exception as e_json_save:
                logger.error(f"Error saving final results JSON: {str(e_json_save)}", exc_info=True)
                simplified_result_file = os.path.join(agent_config_args.log_folder, task_id, 'result_simplified.txt')
                with open(simplified_result_file, 'w', encoding='utf-8') as f:
                    f.write(str(final_json))
        
        return agent_run_results # Return the structured results

    except Exception as e:
        logger.critical(f"Critical error in main execution for task {task_id}: {str(e)}", exc_info=True)
        if task_id:
            error_file_path = os.path.join(agent_config_args.log_folder if agent_config_args else "log", task_id, 'error.json')
            os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
            error_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type
            }
            try:
                with open(error_file_path, 'w', encoding='utf-8') as f:
                    json.dump(error_json, f, indent=4)
            except Exception as e_err_save:
                 logger.error(f"Failed to save error JSON: {str(e_err_save)}")
        raise
    finally:
        if playwright_manager:
            try: await playwright_manager.close()
            except Exception as e_close: logger.error(f"Error closing playwright: {str(e_close)}", exc_info=True)
        
        # Clean up logger handlers for the current task_id
        # This is important in batch mode to avoid duplicate logs or closed file handlers
        current_task_logger = logging.getLogger(task_id or "default_task")
        if current_task_logger and current_task_logger.handlers:
            for handler in list(current_task_logger.handlers): # Iterate over a copy
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                current_task_logger.removeHandler(handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebShop evaluation with tree search")    
    parser.add_argument("--starting-url", type=str, default="http://54.224.220.64:3000/fixed_0",
                      help="Starting URL for the web agent (default: http://54.224.220.64:3000/fixed_0)")
    parser.add_argument("--agent-type", type=str, default="WebShopTreeSearchAgent",
                      help="Type of agent to use (default: WebShopTreeSearchAgent)")
    parser.add_argument("--goal", type=str, default=WEBSHOP_GOAL,
                      help="Goal for the web agent to accomplish")    
    parser.add_argument("--images", type=str, default="",
                      help="Comma-separated paths to image files (e.g., 'path1.jpg,path2.jpg')")    
    parser.add_argument("--task-id", type=str, default=None,
                      help="Task ID for this evaluation run")
    parser.add_argument("--batch-start", type=int, default=None,
                      help="Start task number for batch evaluation")
    parser.add_argument("--batch-end", type=int, default=None,
                      help="End task number for batch evaluation")
    parser.add_argument("--browser-mode", type=str, default="browserbase", choices=["browserbase", "chromium"],
                      help="Browser mode to use (default: browserbase)")
    parser.add_argument("--headless", action="store_true",
                      help="Run in headless mode (default: False, only applicable to chromium)")
    parser.add_argument("--timeout", type=int, default=10,
                      help="Maximum runtime in minutes before terminating (default: 10)")
    
    # Dynamically add AgentConfig fields as arguments
    # Store args that have defaults in AgentConfig to handle them specially
    agent_config_defaults = {
        F.name: (
            F.default if F.default is not dataclasses.MISSING else 
            F.default_factory() if F.default_factory is not dataclasses.MISSING else
            None  # Fallback for fields with no default
        ) 
        for F in AgentConfig.__dataclass_fields__.values() 
    }

    for F in AgentConfig.__dataclass_fields__.values():
        arg_name = f"--{F.name.replace('_', '-')}"
        if any(opt_str == arg_name for action in parser._actions for opt_str in action.option_strings):
            continue # Skip if we manually defined it (like browser-mode, headless, timeout)
        
        current_default = agent_config_defaults.get(F.name)
        
        # Handle boolean fields differently - don't specify 'type' for store_true/store_false actions
        if F.type is bool:
            parser.add_argument(
                arg_name,
                action='store_true' if current_default is False else 'store_false' if current_default is True else 'store',
                help=f"AgentConfig: {F.name} (AgentConfig default: {current_default})"
            )
        else:
            # For non-boolean types
            parser.add_argument(
                arg_name, 
                type=F.type if F.type in (str, int, float) else str,
                action='store',
                help=f"AgentConfig: {F.name} (AgentConfig default: {current_default})"
            )

    args = parser.parse_args()
    
    # Create AgentConfig object: starts with dataclass defaults
    agent_config = AgentConfig()
    
    # Override with command-line arguments IF they were actually provided
    # vars(args) gives a dict of arg_name: value. We only want to set attributes on agent_config
    # if the user explicitly passed the argument, or if argparse has a default value (which we minimized above).
    # For flags (like --headless), if present, args.headless will be True, else False by our parser setup.
    # For other args, if not passed, they might be None or the default from add_argument (which we aim to avoid setting directly).
    
    for arg_name_hyphen, arg_value in vars(args).items():
        attr_name_underscore = arg_name_hyphen.replace('-', '_')
        # Only update if the attribute exists in AgentConfig
        if hasattr(agent_config, attr_name_underscore):
            # Special handling for list-type args that might come as comma-separated strings
            # Example: features. AgentConfig expects a list.
            field_type = AgentConfig.__annotations__.get(attr_name_underscore)
            
            # If arg_value is None, it means the arg was not provided by the user and our parser didn't assign a default.
            # In this case, we want AgentConfig's own default (from __init__ or default_factory) to prevail.
            # So, we only setattr if arg_value is NOT None.
            # However, for boolean flags set by action='store_true'/'store_false', arg_value will be True/False, not None.
            # We need to ensure these are set.
            
            # Check if the argument was actually provided or is a boolean flag
            # This is a bit tricky with argparse. A common way is to set a unique default and check against it.
            # Or, rely on the fact that non-provided, non-boolean args without explicit defaults in add_argument are None.
            if arg_value is not None: # This covers boolean flags and provided arguments
                if isinstance(field_type, type(List)) or (hasattr(field_type, '__origin__') and field_type.__origin__ is list):
                    if isinstance(arg_value, str):
                        setattr(agent_config, attr_name_underscore, [item.strip() for item in arg_value.split(',')])
                    elif isinstance(arg_value, list): # If already a list (e.g. from a default in parser)
                         setattr(agent_config, attr_name_underscore, arg_value)
                    # If it's not a string to be split, and not a list, but AgentConfig expects a list, this might be an issue.
                    # For now, this handles comma-separated strings for list fields.
                elif field_type is bool: # Already handled by action='store_true'/'store_false'
                     setattr(agent_config, attr_name_underscore, arg_value)
                else: # For other types (str, int, float)
                    setattr(agent_config, attr_name_underscore, arg_value)
            # If arg_value is None, we do nothing, letting AgentConfig's default stand.
            
    # Ensure log_folder from AgentConfig is used for task-specific logging
    # This was slightly off before, as agent_config.log_folder might not be set if --log-folder wasn't passed
    # Now, agent_config will have its default for log_folder if not overridden.

    # Process images list from comma-separated string (specific to this script's --images arg)
    images_list = [img.strip() for img in args.images.split(',')] if args.images else []
    
    # Setup root logger for batch overview, if not already set
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.batch_start is not None and args.batch_end is not None:
        logging.info(f"Starting batch evaluation from task {args.batch_start} to {args.batch_end}")
        base_url = args.starting_url.rstrip('/')
        if '_' in base_url: base_url = base_url.rsplit('_', 1)[0]
        
        for task_num in range(args.batch_start, args.batch_end + 1):
            current_task_id = f"webshop_task_{task_num}"
            # Create a *copy* of the main agent_config for this task to avoid interference if log_folder is changed per task
            task_specific_agent_config = dataclasses.replace(agent_config) 
            # It's generally better if the agent internally uses a subfolder of the main log_folder based on task_id
            # For now, assuming the main log_folder is sufficient, or agent handles sub-logging.
            
            task_url = f"{base_url}_{task_num}"
            logging.info(f"\n--- Running WebShop Task {task_num} ({current_task_id}) at {task_url} ---")
            try:
                asyncio.run(main(
                    task_url,
                    args.agent_type,
                    args.goal,
                    images_list,
                    current_task_id,
                    task_specific_agent_config, 
                    args.timeout
                ))
                logging.info(f"--- Completed Task {task_num} ({current_task_id}) ---")
            except Exception as e:
                logging.error(f"--- Error in Task {task_num} ({current_task_id}): {str(e)} ---", exc_info=True)
                continue
        logging.info("Batch evaluation finished.")
    else:
        current_task_id = args.task_id or f"webshop_tree_search_task_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting single task evaluation: {current_task_id}")
        asyncio.run(main(
            args.starting_url,
            args.agent_type,
            args.goal,
            images_list,
            current_task_id,
            agent_config, # Use the global agent_config for single tasks
            args.timeout
        ))
        logging.info(f"Single task evaluation {current_task_id} finished.") 