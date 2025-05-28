import inspect
from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, humanevalpack,
               instruct_humaneval, instruct_wizard_humaneval, mbpp, multiple,
               parity, python_bugs, quixbugs, recode, santacoder_fim)

from .private_algorithmic import ( humaneval_translate, humaneval_complete, robust_mbpp_complete,
                                  humaneval_implementation, mbpp_implementation, humaneval_bugfix, mbpp_bugfix,robust_mbpp_translate,robust_mbpp_generate_prompt,robust_mbpp_generate_instruct,robust_mbpp_generate_preprocess)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    "instruct_wizard_humaneval": instruct_wizard_humaneval.HumanEvalWizardCoder,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
    **santacoder_fim.create_all_tasks(),
    **humaneval_translate.create_all_tasks(),
    **humaneval_complete.create_all_tasks(),
    **humaneval_implementation.create_all_tasks(),
    **mbpp_implementation.create_all_tasks(),
    **humaneval_bugfix.create_all_tasks(),
    **mbpp_bugfix.create_all_tasks(),
    **robust_mbpp_translate.create_all_tasks(),
    **robust_mbpp_complete.create_all_tasks(),
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
