"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""


from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import extract_python_code

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


IMPORT_HELPER = [
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import statistics",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
]

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": create_task(True), "humaneval-unstripped": create_task(False)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self, model_series, model_type, stop_words, **kwargs):
            super().__init__(strip_prompt, model_series=model_series, model_type=model_type, 
                             stop_words=stop_words, **kwargs)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "openai_humaneval"
    DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/instructhumaneval"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=10.0, 
                 model_series="", model_type="causal_base", stop_words=[]):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.model_series = model_series
        self.model_type = model_type
        self.use_instruction = False
        self.instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the python code for this function.\n"
        if self.model_type == "causal_chat":
            self.use_instruction = True
            self.stop_words = ["if __name__", "\nprint", "\nclass"]
        self.stop_words += stop_words

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = doc["prompt"]
        if self.strip_prompt:
            prompt = prompt.strip()
        if self.use_instruction:
            return self.instruction + prompt
        else:
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        context = self.dataset["test"]["context"][idx]
        generation = generation[len(prompt) :]

        ## 回复的直接是续写
        if (not self.use_instruction) or generation.startswith("\n    "):
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = context + generation
        else:
            generation = extract_python_code(generation)
            generation = self._stop_at_stop_token(generation, self.stop_words)
            ## 回复的是代码块并且是续写 
            if generation.startswith("\n    ") or generation.startswith("    "):
                if not generation.startswith("\n"):
                    generation = "\n" + generation
                generation = context + generation
            ## 回复的是代码块不是续写
            else:
                sep_index = generation.find(":\n    ") + 1
                if generation[sep_index :].startswith("\n        "):
                    generation = generation.lstrip()
                else:
                    generation = generation[sep_index :]
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation

        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        python_imports = "\n".join(IMPORT_HELPER)
        generations = [
            [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
        ]

        metrics, cases = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
            task="humaneval",
        )

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(references)
        sum_ = metrics["pass@1"] * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = metrics["pass@1"]
        stat["max"] = metrics["pass@1"]
        stat["mean"] = metrics["pass@1"]
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        stats = [[stat, "humaneval_zero_shot-generation-generation:"]]

        return metrics, cases, stats
