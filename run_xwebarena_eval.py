import argparse
from dotenv import load_dotenv
load_dotenv()
from lwats.core_async.agent_factory import setup_prompting_web_agent
from lwats.core_async.config import PromptingAgentConfig
from xwebarena_evaluation_suite import evaluator_router, image_utils
import asyncio,json,requests
from PIL import Image


async def main(headless, browser_mode, config_file, agent_type, action_generation_model, plan):
    log_folder = "./log/"+config_file.replace(".json","")+"/"
    model = "gpt-4-mini"
    features = "axtree"
    num_simulations = 100
    exploration_weight = 1.41
    
    cfgfileContent=""
    with open(config_file) as f:
        _c = json.load(f)
        goal = _c["intent"]
        task_id = _c["task_id"]
        starting_url = _c["start_url"]
        storage_state = _c["storage_state"]
        image_paths = _c.get("image", None)   # only for vwa
        if image_paths is not None:
            if isinstance(image_paths, str):
                images_list = [image_paths]
            elif isinstance(image_paths, list):
                images_list = image_paths
            else:
                images_list = []
        else:
            images_list = []
        cfgfileContent += f"[Config file]: {config_file}" + "\n"
        cfgfileContent += f"[Goal]: {goal}" + "\n"
        cfgfileContent += f"[Starting Url]: {starting_url}" + "\n"
        cfgfileContent += f"[Storage State]: {storage_state}" + "\n"
        cfgfileContent += f"[Task id]: {task_id}" + "\n"
        cfgfileContent += f"[len(images_list)]: {len(images_list)}" + "\n"

    # Create config
    config = PromptingAgentConfig(
        headless=headless,
        browser_mode=browser_mode,
        storage_state=storage_state,
        action_generation_model=action_generation_model,
        features=['axtree'],
        branching_factor=5,
        log_folder=log_folder,
        fullpage=True
    )
    
    agent, playwright_manager = await setup_prompting_web_agent(
        starting_url=starting_url,
        goal=goal,
        images=images_list,
        agent_type=agent_type,
        config=config
    )
    
    # Ensure the agent is of the specified type
    expected_agent_class = globals().get(agent_type)
    if expected_agent_class and not isinstance(agent, expected_agent_class):
        raise TypeError(f"Agent is not an instance of {agent_type}")
    
    # Run the search
    trajectory, result = await agent.send_prompt(plan if plan is not None else goal)

    evaluator = evaluator_router(
        config_file, captioning_fn=None
    )
    score = evaluator(    # TODO: score not always working
        trajectory=trajectory,
        config_file=config_file,
        page=playwright_manager.page
    )
    
    # Close the playwright_manager when done
    await playwright_manager.close()

    res="******************************config_file******************************" + "\n" + str(cfgfileContent) + "\n" +    \
        "******************************trajectory******************************" +"\n" + str(trajectory) + "\n"+    \
        "******************************result******************************" + "\n" + str(result) + "\n" +    \
        "******************************score******************************" + "\n" + str(score) + "\n"    
    print(res)
    with open(log_folder+"output.txt", "w") as f:
        f.write(res)


'''
python run_xwebarena.py --browser-mode chromium --agent-type "PromptAgent" --action_generation_model "gpt-4o-mini" --config-file ./xwebarena_evaluation_suite/configs/wa_117.json
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
    asyncio.run(main(args.headless,
        args.browser_mode,
        args.config_file,
        args.agent_type,
        args.action_generation_model,
        args.plan))
