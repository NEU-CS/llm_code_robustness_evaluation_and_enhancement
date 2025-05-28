"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file
from bigcode_eval.utils import extract_code

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]

LANGUAGE_TO_EXTENSION = {
    "py": "python",
    "cpp": "cpp",
    "js": "javascript",
    "java": "java",
    "sh": "shell",
}

IMPORT_HELPER = {
    "py": [
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
    ],
}

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {f"multiple-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self, model_series, model_type, stop_words):
            super().__init__(language, model_series, model_type, stop_words)

    return MultiPLE


class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/MultiPL-E"
    DATASET_NAME = None

    def __init__(self, language, model_series, model_type, stop_words):
        self.language = language
        # self.DATASET_NAME = f"humaneval-{language}"
        self.DATASET_NAME = language
        self.model_series = model_series
        self.model_type = model_type
        # we need the dataset to get stop words for each language
        # self.dataset = load_dataset(
        #     GeneralMultiPLE.DATASET_PATH,
        #     self.DATASET_NAME,
        #     revision=self.DATASET_REVISION)
        self.dataset = load_from_disk(self.DATASET_PATH + "/" + self.DATASET_NAME)
        self.dataset = self.dataset.shuffle(seed=0)
        stop_tokens = self.dataset["test"][0]["stop_tokens"]
        self.use_instruction = False
        self.instruction = f"Read the following function signature and docstring, and fully implement the function described. Your response should only contain the {LANGUAGE_TO_EXTENSION[language]} code for this function.\n"
        if self.model_type == "causal_chat":
            self.use_instruction = True
            if self.language == "py":
                stop_tokens = ["\nclass"]
            elif self.language == "java":
                stop_tokens = []
            elif self.language == "js":
                stop_tokens = ["\nconsole.log"]
        super().__init__(
            stop_words=stop_tokens,
            requires_execution=True,
        )
        self.stop_words += stop_words

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = doc["prompt"].strip()
        if self.use_instruction:
            return self.instruction + prompt
        else:
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["tests"]

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        context = self.dataset["test"]["prompt"][idx].strip()
        generation = generation[len(prompt) :]

        if (not self.use_instruction) or (generation.startswith("\n  ") and self.language != "java"):
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = context + generation
        else:
            generation = extract_code(generation, self.language)
            generation = self._stop_at_stop_token(generation, self.stop_words)
            if self.language == "py":
                if generation.startswith("\n    ") or generation.startswith("    "):
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
                else:
                    sep_index = generation.find(":\n    ") + 1
                    if generation[sep_index :].startswith("\n        "):
                        generation = generation.lstrip()
                    else:
                        generation = generation[sep_index :]
                        if not generation.startswith("\n"):
                            generation = "\n" + generation
                        generation = context + generation
            elif self.language in ["cpp", "sh", "js"]:
                if generation.startswith("\n  ") or generation.startswith("  "):
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
                else:
                    sep_index = generation.find("{\n  ") + 1
                    generation = generation[sep_index :]
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
            elif self.language == "java":
                if generation.startswith("\n        ") or generation.startswith("        "):
                    generation = self._stop_at_stop_token(generation, ["\n    }"])
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
                elif generation.startswith("\n    ") or generation.startswith("    "):
                    generation = self._stop_at_stop_token(generation, ["\n}"])
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
                else:
                    sep_index = generation.find(") {\n    ")
                    if sep_index != -1 and ("class" in generation or "public" in generation):
                        sep_index += 3
                        if generation[sep_index :].startswith("\n        "):
                            generation = self._stop_at_stop_token(generation, ["\n    }"])
                        else:
                            generation = self._stop_at_stop_token(generation, ["\n}"])
                        generation = generation[sep_index :]
                        generation = context + generation
                    else:
                        if not generation.startswith("\n"):
                            generation = "\n" + generation.rstrip("}")
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
        if self.language == "py":
            python_imports = "\n".join(IMPORT_HELPER["py"])
            generations = [
                [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
            ]

        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        for file_name in os.listdir(temp_dir):
            if ".results.json" in file_name:
                os.remove(Path(temp_dir, file_name))
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 100], result)
            if k <= len(generations[0])
        }

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(references)
        sum_ = results["pass@1"] * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = results["pass@1"]
        stat["max"] = results["pass@1"]
        stat["mean"] = results["pass@1"]
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        stats = [[stat, f"multiple_zero_shot:{self.DATASET_NAME}-generation-generation,"]]

        return results, None, stats
