import argparse
from dotenv import load_dotenv
load_dotenv()
from lwats.core_async.agent_factory import setup_prompting_web_agent
from lwats.core_async.config import PromptingAgentConfig
import asyncio
import os
import json
import datetime
import logging

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

async def main(headless, browser_mode, starting_url, agent_type, goal, 
         action_generation_model, images, plan, task_id=None):
    """
    Main function to run the WebShop evaluation.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        browser_mode (str): Browser mode (chromium/browserbase)
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use
        goal (str): Task goal/instruction
        action_generation_model (str): Model to use for action generation
        images (str): Comma-separated list of image paths
        plan (str): Optional plan to follow
        task_id (str): Optional task ID for logging
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

    # Split the comma-separated string into a list of images
    images_list = [img.strip() for img in images.split(',')] if images else []
    
    try:
        # Create config
        config = PromptingAgentConfig(
            headless=headless,
            browser_mode=browser_mode,
            storage_state=None,  # No storage state for WebShop
            action_generation_model=action_generation_model,
            features=['axtree'],
            branching_factor=5,
            log_folder=log_folder if task_id else "log",
            fullpage=True,
            account_reset=False
        )
        
        agent, playwright_manager = await setup_prompting_web_agent(
            starting_url=starting_url,
            goal=goal,
            images=images_list,
            agent_type=agent_type,
            config=config
        )
        
        # Run the search
        trajectory, result = await agent.send_prompt(plan if plan is not None else goal)
        logger.info("Trajectory:")
        logger.info(trajectory)
        logger.info("Result:")
        logger.info(result)
        
        # Save results
        if task_id:
            result_file = os.path.join(log_folder, 'result.json')
            final_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "trajectory": trajectory,
                "result": result,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type,
                "action_generation_model": action_generation_model,
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=4)
        
        # Close the playwright_manager when done
        await playwright_manager.close()
        return trajectory, result

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
    parser = argparse.ArgumentParser(description="Run WebShop evaluation")
    parser.add_argument("--headless", type=bool, default=False,
                        help="Specify if the browser should run in headless mode (default: False)")
    parser.add_argument("--browser-mode", type=str, default="chromium",
                        help="Specify the browser mode (default: chromium)")
    parser.add_argument("--starting-url", type=str, default="http://54.224.220.64:3000/fixed_0",
                        help="Starting URL for the web agent (default: http://54.224.220.64:3000/fixed_0)")
    parser.add_argument("--agent-type", type=str, default="PromptAgent",
                        help="Type of agent to use (default: PromptAgent)")
    parser.add_argument("--goal", type=str, default=WEBSHOP_GOAL,
                        help="Goal for the web agent to accomplish")
    parser.add_argument("--action_generation_model", type=str, default="gpt-4o",
                        help="Action generation model (default: gpt-4o)")
    parser.add_argument("--images", type=str, default="",
                        help="Comma-separated paths to image files (e.g., 'path1.jpg,path2.jpg')")
    parser.add_argument("--plan", type=str, default=None,
                        help="Optional plan for the web agent to follow (default: None)")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Task ID for this evaluation run")
    parser.add_argument("--batch-start", type=int, default=None,
                        help="Start task number for batch evaluation")
    parser.add_argument("--batch-end", type=int, default=None,
                        help="End task number for batch evaluation")
    
    args = parser.parse_args()

    # Handle batch evaluation
    if args.batch_start is not None and args.batch_end is not None:
        base_url = args.starting_url.rstrip('/')
        # Get the base URL before the task number
        if '_' in base_url:
            base_url = base_url.rsplit('_', 1)[0]
        
        for task_num in range(args.batch_start, args.batch_end + 1):
            task_url = f"{base_url}_{task_num}"
            task_id = f"webshop_task_{task_num}"
            print(f"\nRunning WebShop task {task_num} at {task_url}")
            try:
                trajectory, result = asyncio.run(main(
                    args.headless,
                    args.browser_mode,
                    task_url,
                    args.agent_type,
                    args.goal,
                    args.action_generation_model,
                    args.images,
                    args.plan,
                    task_id
                ))
                print(f"Completed task {task_num}")
            except Exception as e:
                print(f"Error in task {task_num}: {str(e)}")
                continue
    else:
        # Single task evaluation
        task_id = args.task_id or f"webshop_task_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trajectory, result = asyncio.run(main(
            args.headless,
            args.browser_mode,
            args.starting_url,
            args.agent_type,
            args.goal,
            args.action_generation_model,
            args.images,
            args.plan,
            task_id
        )) 