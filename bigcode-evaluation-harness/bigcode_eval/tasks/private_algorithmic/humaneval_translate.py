"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""
import json
import re
import os
from collections import Counter, defaultdict
from datasets import load_from_disk
from evaluate import load

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import remove_after_return

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import extract_code

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
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

IMPORT_HELPER = {
    "python": [
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
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "using namespace std;",
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
    "java": [
        "import java.util.*;",
        "import java.lang.*;"
    ],
    "javascript": []
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "javascript": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}




LANGUAGES = ["python", "cpp", "javascript", "java", "go", "rust"]

def create_all_tasks():
    return {f"humaneval_translate-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvaltrans(GeneralHumanEvaltrans):
        def __init__(self):
            super().__init__(language)

    return HumanEvaltrans


class GeneralHumanEvaltrans(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "openai_humaneval"

    def __init__(self, language, k=[1, 10, 100], model_series="", model_type="causal_base"):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )
        self.language = language
        self.k = k
        self.model_series = model_series
        self.model_type = model_type
        self.DATASET_PATH = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/translate/humaneval_{language}.json"
        self.instruction = ""

    def get_dataset(self):
        self.dataset = json.load(open(self.DATASET_PATH, "r"))
        # dataset = []
        # for item in data:
        #     dataset.append(item["code_str_rename_del_oneRow"])
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        original_code = doc["python_canonical_solution"]
        declaration = doc["python_declaration"]
        head = doc[f"{self.language}_declaration"]
        function_head = declaration.strip().split('\n')[-1]
        prompt = f"请你将已有的python程序\n```python\n{head}\n{original_code}\n```\n改写成{self.language}的代码,\n{self.language}语言对应的函数头为{function_head}，请你返回包含此函数头的完整代码\n"
        return prompt   

    def get_reference(self, doc):
        test_func = doc[f"{self.language}_test"]
        return "\n" + test_func + "\n" 

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        # prompt = self.get_prompt(self.dataset["test"][idx])
        # context = self.dataset["test"]["context"][idx]
        # generation = generation[len(prompt) :]
        prompt = self.get_prompt(self.dataset[idx])
        generation = generation[len(prompt):]
        generation = extract_code(generation, self.language)
        return generation
        

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("./bigcode_eval/tasks/custom_metrics/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.language]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.language]
        import_helper = "\n".join(IMPORT_HELPER[self.language])
        generations = [
           [(import_helper + "\n" + g).strip() for g in gen] for gen in generations
           ]

        metrics, cases = code_metric.compute(
            references=references,
            predictions=generations,
            language= self.language,
            timeout=timeout,
            num_workers=num_workers,
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
        stats = [[stat, "humaneval_translate_zero_shot-generation-generation:"]]

        return metrics, cases, stats
