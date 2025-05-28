"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import json

from evaluate import load

from bigcode_eval.base import Task

_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["introductory", "interview", "competition"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"apps-{level}": create_task(level) for level in LEVELS}


def create_task(level):
    class APPS(GeneralAPPS):
        def __init__(self, model_series, **kwargs):
            super().__init__(level, model_series=model_series, **kwargs)

    return APPS


class GeneralAPPS(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "codeparrot/apps"
    DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/apps"
    DATASET_NAME = None

    def __init__(self, level, model_series, k_list=[1, 10, 100]):
        self.model_series = model_series
        self.DATASET_NAME = level
        super().__init__(
            stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
        self.k_list = k_list

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None

        prompt = "QUESTION:\nWrite a python code to solve the following coding problem that obeys the constraints and passes the example test cases. "
        prompt += "Please wrap your code answer using ```:\n"
        prompt += doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n```"
        return prompt

        # prompt = "Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases."
        # prompt += "\nQUESTION:\n"
        # prompt += doc["question"]
        # if starter_code:
        #     prompt += starter_code
        # if not fn_name:
        #     call_format = "\nUse Standard Input format"
        #     prompt += call_format
        # else:
        #     call_format = "\nUse Call-Based format"
        #     prompt += call_format
        # prompt += "\nANSWER:\n"
        # return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return None

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for APPS)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        generation = generation.lstrip("python")
        try:
            generation = generation.split("```", 1)[0]
        except IndexError:
            # happens when prompts were very long and got truncated
            pass
        return generation
        # try:
        #     generation = generation.split("\nANSWER:", 1)[1]
        # except IndexError:
        #     # happens when prompts were very long and got truncated
        #     pass
        # return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        # code_metric = load("codeparrot/apps_metric")
        code_metric = load("./bigcode_eval/tasks/custom_metrics/apps_metric")
        # if level is None:
        #     level = self.DATASET_NAME
        results = code_metric.compute(
            predictions=generations, k_list=self.k_list, level=self.DATASET_NAME
        )

        if results["pass_at_k"]:
            metric = results["pass_at_k"]["pass@1"]
        else:
            metric = results["strict_accuracy"]

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(references)
        sum_ = metric * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = metric
        stat["max"] = metric
        stat["mean"] = metric
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        stats = [[stat, f"apps_zero_shot:{self.DATASET_NAME}-generation-generation,"]]

        return results, None, stats
