import inspect
from pprint import pprint
from .private_algorithmic import (robust_mbpp_generate_prompt,robust_mbpp_generate_instruct,robust_mbpp_generate_preprocess)

TASK_REGISTRY = {
    **robust_mbpp_generate_prompt.create_all_tasks(),
    **robust_mbpp_generate_instruct.create_all_tasks(),
    **robust_mbpp_generate_preprocess.create_all_tasks()
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        # if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
        #     kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            if args.load_data_path:
                kwargs["load_data_path"] = args.load_data_path
            else:
                kwargs["load_data_path"] = args.save_generations_path
        if "model_series" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["model_series"] = args.model_series
        if "model_type" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["model_type"] = args.model_type
        if "stop_words" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["stop_words"] = args.stop_words
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
