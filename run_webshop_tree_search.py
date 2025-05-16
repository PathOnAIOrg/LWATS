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

# Custom JSON encoder to handle non-serializable objects
class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle LATSNode serialization
        if hasattr(obj, '__dict__'):
            # Convert object to dictionary, but filter out any non-serializable values
            result = {}
            for key, value in obj.__dict__.items():
                try:
                    # Test if the value is serializable
                    json.dumps(value)
                    result[key] = value
                except (TypeError, OverflowError):
                    # If not serializable, convert to string representation
                    result[key] = str(value)
            return result
        # Handle other non-serializable types
        elif hasattr(obj, 'isoformat'):  # For datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, list, dict)):
            return list(obj)  # Convert iterables to lists
        # If we can't handle it, let the parent class handle it
        return super().default(obj)

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

async def main(starting_url, agent_type, goal, images, task_id=None, agent_config=None):
    """
    Main function to run the WebShop evaluation with tree search.
    
    Args:
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use
        goal (str): Task goal/instruction
        images (list): List of image paths
        task_id (str): Optional task ID for logging
        agent_config (AgentConfig): Configuration for the agent
    """
    # Setup logging
    if task_id:
        log_folder = os.path.join("log", task_id)
        os.makedirs(log_folder, exist_ok=True)
        logger, log_fh, console_handler = setup_logger(task_id, log_folder)
        logger.info(f"Starting evaluation for task {task_id}")
        logger.info(f"Starting URL: {starting_url}")
    else:
        logger = logging.getLogger("default")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        log_fh = None
    
    try:
        # Setup search agent with tree search capability
        agent, playwright_manager = await setup_search_agent(
            agent_type=agent_type,
            starting_url=starting_url,
            goal=goal,
            images=images,
            agent_config=agent_config
        )
        
        # ***Print and open browserbase URL if using browserbase mode***
        if agent_config.browser_mode == "browserbase":
            live_browser_url = await playwright_manager.get_live_browser_url()
            session_id = await playwright_manager.get_session_id()
            if live_browser_url:
                logger.info(f"✅ BrowserBase session created!")
                logger.info(f"🔍 You can view the browser actions at: {live_browser_url}")
                logger.info(f"📝 Session ID: {session_id}")
                print(f"\n✅ BrowserBase session created!")
                print(f"🔍 You can view the browser actions at: {live_browser_url}")
                print(f"📝 Session ID: {session_id}")
                
                # Automatically open the URL in the default browser
                print("Opening debugger URL in your default browser...")
                webbrowser.open(live_browser_url)
        
        # Run the search
        results = await agent.run()
        
        # Log and save results
        logger.info("Results:")
        try:
            # Use custom encoder for non-serializable objects
            logger.info(json.dumps(results, indent=2, cls=NodeEncoder))
        except Exception as e:
            logger.error(f"Error serializing results: {str(e)}")
            logger.info(f"Raw results: {str(results)}")
        
        # Save results
        if task_id:
            result_file = os.path.join(log_folder, 'result.json')
            final_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "results": results,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type
            }
            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, indent=4, cls=NodeEncoder)
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
                # Save a simplified version
                with open(result_file, 'w', encoding='utf-8') as f:
                    simplified_json = {
                        "task_id": task_id,
                        "goal": goal,
                        "starting_url": starting_url,
                        "results_summary": str(results),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "agent_type": agent_type
                    }
                    json.dump(simplified_json, f, indent=4)
        
        # Close the playwright_manager when done
        await playwright_manager.close()
        return results

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if task_id:
            result_file = os.path.join(log_folder, 'error.json')
            error_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_json, f, indent=4)
        raise
    finally:
        if log_fh:
            logger.removeHandler(log_fh)
        logger.removeHandler(console_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebShop evaluation with tree search")    
    parser.add_argument("--starting-url", type=str, default="http://54.224.220.64:3000/fixed_0",
                      help="Starting URL for the web agent (default: http://54.224.220.64:3000/fixed_0)")
    parser.add_argument("--agent-type", type=str, default="LATSAgent",
                      help="Type of agent to use (default: LATSAgent)")
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
    parser.add_argument("--browser-mode", type=str, default="browserbase",
                      help="Browser mode to use: 'browserbase' or 'chromium' (default: browserbase)")
    parser.add_argument("--headless", action="store_true",
                      help="Run in headless mode (default: False)")
    
    # Add some common agent config arguments directly
    parser.add_argument("--branching-factor", type=int, default=5,
                      help="Branching factor for tree search (default: 5)")
    parser.add_argument("--max-depth", type=int, default=3,
                      help="Maximum depth for tree search (default: 3)")
    parser.add_argument("--storage-state", type=str, default='state.json',
                      help="Storage state file (default: state.json)")
    parser.add_argument("--action-generation-model", type=str, default="gpt-4o",
                      help="Action generation model (default: gpt-4o)")
    
    args = parser.parse_args()
    
    # Create default agent config
    agent_config = AgentConfig()
    
    # Override config with parsed args where they exist
    for key, value in vars(args).items():
        if value is not None and hasattr(agent_config, key.replace('-', '_')):
            setattr(agent_config, key.replace('-', '_'), value)
    
    # Process images list from comma-separated string
    images_list = [img.strip() for img in args.images.split(',')] if args.images else []
    
    # Handle batch evaluation
    if args.batch_start is not None and args.batch_end is not None:
        base_url = args.starting_url.rstrip('/')
        # Get the base URL before the task number
        if '_' in base_url:
            base_url = base_url.rsplit('_', 1)[0]
        
        for task_num in range(args.batch_start, args.batch_end + 1):
            task_url = f"{base_url}_{task_num}"
            task_id = f"webshop_tree_search_task_{task_num}"
            print(f"\nRunning WebShop task {task_num} at {task_url}")
            try:
                results = asyncio.run(main(
                    task_url,
                    args.agent_type,
                    args.goal,
                    images_list,
                    task_id,
                    agent_config
                ))
                print(f"Completed task {task_num}")
            except Exception as e:
                print(f"Error in task {task_num}: {str(e)}")
                continue
    else:
        # Single task evaluation
        task_id = args.task_id or f"webshop_tree_search_task_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = asyncio.run(main(
            args.starting_url,
            args.agent_type,
            args.goal,
            images_list,
            task_id,
            agent_config
        )) 