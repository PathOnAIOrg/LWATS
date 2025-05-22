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
from playwright.async_api import async_playwright

async def extract_instructions_from_webpage(url, browser_mode="chromium"):
    """
    Extract task instructions from the WebShop page.
    
    Args:
        url (str): URL of the WebShop task page
        browser_mode (str): Browser engine to use (chromium/browserbase)
        
    Returns:
        str: Extracted instructions text
    """
    async with async_playwright() as p:
        # Choose browser engine based on browser_mode parameter
        if browser_mode.lower() == "browserbase":
            browser = await p.firefox.launch(headless=True)
        else:  # Default to chromium
            browser = await p.chromium.launch(headless=True)
            
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle")
            
            # Try multiple different selectors that might contain the instruction
            # Since we don't know the exact structure, we'll try several common patterns
            possible_selectors = [
                "div.instruction-text", 
                "div.instruction", 
                "div.task-instruction",
                "div.description",
                "div:has-text('Instruction:')",
                "div:has-text('Task:')",
                "div.container div h3:has-text('Instruction') + div",
                "div.container div h4:has-text('Instruction') + div",
                "p.instruction"
            ]
            
            instruction_text = None
            for selector in possible_selectors:
                try:
                    # Use a shorter timeout for each individual selector attempt
                    element = await page.wait_for_selector(selector, timeout=1000, state="visible")
                    if element:
                        instruction_text = await element.inner_text()
                        if instruction_text and len(instruction_text.strip()) > 10:  # Ensure we got meaningful text
                            instruction_text = instruction_text.strip()
                            break
                except Exception:
                    continue
            
            # If we found text using selectors, return it
            if instruction_text:
                # Clean up the text
                instruction_text = clean_instruction_text(instruction_text)
                return instruction_text
            
            # Fallback: Look for text on the page containing common instruction indicators
            content = await page.content()
            
            # Try to find common instruction markers in the page content
            instruction_markers = ["Instruction:", "Task:", "Your task is", "You need to", "Please find"]
            
            for marker in instruction_markers:
                if marker in content:
                    # Get the page text rather than HTML - more reliable for extraction
                    all_text = await page.evaluate("() => document.body.innerText")
                    
                    # Find the marker in the text
                    start_idx = all_text.find(marker)
                    if start_idx >= 0:
                        # Extract from marker to the next double newline (paragraph break)
                        # or up to 500 chars max
                        start_idx += len(marker)
                        end_idx = all_text.find("\n\n", start_idx)
                        if end_idx == -1 or end_idx - start_idx > 500:
                            end_idx = start_idx + 500
                        
                        extracted_text = all_text[start_idx:end_idx].strip()
                        if len(extracted_text) > 10:  # Ensure we got meaningful text
                            # Clean up the text
                            extracted_text = clean_instruction_text(extracted_text)
                            return extracted_text
                        
            # Final fallback: Take a screenshot for debugging and return a generic message
            await page.screenshot(path="webshop_page.png")
            
            # Check if there's a search interface, which likely means it's the WebShop
            search_box = await page.query_selector("input[type='search'], input[placeholder*='search']")
            if search_box:
                return "Explore the WebShop interface and complete the shopping task based on the product requirements shown on the page. Search for appropriate items, navigate through results, select the correct product with matching requirements, and complete the purchase to get your score."
            
            return "Could not extract specific instructions from the page. Please proceed with the task as displayed in the browser."
        finally:
            await browser.close()

def clean_instruction_text(text):
    """
    Clean up the extracted instruction text by removing common prefixes and suffixes.
    
    Args:
        text (str): Raw instruction text
        
    Returns:
        str: Cleaned instruction text
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "WebShop\nInstruction:\n",
        "WebShop\nInstruction:",
        "Instruction:\n",
        "Instruction:",
        "Task:\n",
        "Task:"
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Remove common suffixes
    suffixes_to_remove = [
        "\nSearch",
        "\nSearch:",
        "\nFind:",
        "\nClick"
    ]
    
    for suffix in suffixes_to_remove:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
            break
    
    return text.strip()

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

async def main(headless, browser_mode, starting_url, agent_type, goal, 
         action_generation_model, images, plan, task_id=None):
    """
    Main function to run the WebShop evaluation.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        browser_mode (str): Browser mode (chromium/browserbase)
        starting_url (str): Initial URL to start from
        agent_type (str): Type of agent to use
        goal (str): Task goal/instruction (optional, will be extracted from page if None)
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
        
        # Get the score from webshop_score.json (created by PromptAgent)
        score = get_webshop_score(log_folder)
        logger.info(f"Final score: {score}")
        
        # Save results
        if task_id:
            result_file = os.path.join(log_folder, 'result.json')
            final_json = {
                "task_id": task_id,
                "goal": goal,
                "starting_url": starting_url,
                "trajectory": trajectory,
                "result": result,
                "score": score,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_type": agent_type,
                "action_generation_model": action_generation_model,
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=4)
        
        # Close the playwright_manager when done
        await playwright_manager.close()
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
    parser.add_argument("--goal", type=str, default=None,
                        help="Goal for the web agent to accomplish (if not provided, will be extracted from webpage)")
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
                trajectory, result, score = asyncio.run(main(
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
                print(f"Completed task {task_num} with score {score}")
            except Exception as e:
                print(f"Error in task {task_num}: {str(e)}")
                continue
    else:
        # Single task evaluation
        task_id = args.task_id or f"webshop_task_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trajectory, result, score = asyncio.run(main(
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
        print(f"Final score: {score}") 