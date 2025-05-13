import argparse
from dotenv import load_dotenv
load_dotenv()
from lwats.core_async.agent_factory import setup_prompting_web_agent
from xwebarena_evaluation_suite import evaluator_router, image_utils
import asyncio,json



async def main(headless, browser_mode, config_file, agent_type, action_generation_model, plan):
    log_folder = "log"
    model = "gpt-4-mini"
    features = "axtree"
    num_simulations = 100
    exploration_weight = 1.41
    
    print(config_file)
    with open(config_file) as f:
        _c = json.load(f)
        goal = _c["intent"]
        task_id = _c["task_id"]
        starting_url = _c["start_url"]
        storage_state = _c["storage_state"]
        image_paths = _c.get("image", None)   # only for vwa
        images = []
        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                # Load image either from the web or from a local path.
                if image_path.startswith("http"):
                    input_image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    input_image = Image.open(image_path)
                images.append(input_image)
        print(f"[Config file]: {config_file}")
        print(f"[Goal]: {goal}")
        print(f"[Starting Url]: {starting_url}")
        print(f"[Storage State]: {storage_state}")
        print(f"[Task id]: {task_id}")
        print(len(images))
    
    # Split the comma-separated string into a list of images
    images_list = [img.strip() for img in images.split(',')] if images else []
    
    agent, playwright_manager = await setup_prompting_web_agent(
        starting_url=starting_url,
        goal=goal,
        images=images_list,
        agent_type=agent_type,
        features=features,
        branching_factor=5,
        log_folder=log_folder,
        storage_state=storage_state,
        headless=headless,
        browser_mode=browser_mode,
        default_model="gpt-4o-mini",
        planning_model="gpt-4o",
        action_generation_model=action_generation_model,
        action_grounding_model="gpt-4o",
        evaluation_model="gpt-4o",
        fullpage=True,
    )
    
    # Ensure the agent is of the specified type
    expected_agent_class = globals().get(agent_type)
    if expected_agent_class and not isinstance(agent, expected_agent_class):
        raise TypeError(f"Agent is not an instance of {agent_type}")
    
    # Run the search
    trajectory, result = await agent.send_prompt(plan if plan is not None else goal)
    print(trajectory)
    print(result)

    evaluator = evaluator_router(
        config_file, captioning_fn=None
    )
    score = evaluator(
        trajectory=trajectory,
        config_file=config_file,
        page=playwright_manager.page
    )
    print(score)
    
    # Close the playwright_manager when done
    playwright_manager.close()
    return trajectory, result, score


'''
python run_xwebarena.py \
    --browser-mode chromium \
    --config-file ./xwebarena_evaluation_suite/configs/wa_117.json \
    --agent-type "PromptAgent" \
    --action_generation_model "gpt-4o-mini" 
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run web agent with specified configuration")
    parser.add_argument("--config-file", type=str,
                        help="webarena config json file")
    parser.add_argument("--headless", type=bool, default=False,
                        help="Specify if the browser should run in headless mode (default: False)")
    parser.add_argument("--browser-mode", type=str, default="chromium",
                        help="Specify the browser mode (default: chromium)")
    # parser.add_argument("--storage-state", type=str, default=None,
    #                     help="Storage state json file")
    parser.add_argument("--action_generation_model", type=str, default="gpt-4o",
                        help="action grounding model, right now only supports openai models")
    # parser.add_argument("--starting-url", type=str, default="http://xwebarena.pathonai.org:7770/",
    #                     help="Starting URL for the web agent")
    parser.add_argument("--agent-type", type=str, default="LATSAgent",
                        help="Type of agent to use (default: LATSAgent)")
    # parser.add_argument("--goal", type=str, default="search running shoes, click on the first result",
    #                     help="Goal for the web agent to accomplish")
    # parser.add_argument("--images", type=str, default="",
    #                     help="Comma-separated paths to image files (e.g., 'path1.jpg,path2.jpg')")
    parser.add_argument("--plan", type=str, default=None,
                        help="Optional plan for the web agent to follow (default: None)")
    # parser.add_argument("--evaluator-type", type=str, default=None,
    #                     help="Type of evaluator to use (default: None, no evaluation)")
    # parser.add_argument("--eval-url", type=str, default=None,
    #                     help="URL for evaluation purposes")
    # parser.add_argument("--eval-criteria", type=str, default=None,
    #                     help="Criteria for evaluation")
    args = parser.parse_args()
    
    config_file = args.config_file

    # # Run the async main function with asyncio
    trajectory, result, score = asyncio.run(main(args.headless,
        args.browser_mode,
        args.config_file,
        args.agent_type,
        args.action_generation_model,
        args.plan))

    print(trajectory)
    print(result)
    print(score)
