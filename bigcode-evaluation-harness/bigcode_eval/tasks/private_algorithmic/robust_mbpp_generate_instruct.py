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
import random
import sys
import copy
from math import inf
sys.path.append("../perturbation_pipeline")
from pipeline import PerturbationPipeline

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
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
    "java": [
        "import java.util.*;",
        "import java.util.OptionalInt;",
        "import java.util.stream.IntStream;",
        "import java.util.stream.Collectors;",
        "import java.util.regex.Matcher;",
        "import java.util.regex.Pattern;",
        "import java.util.Arrays;",
        "import java.util.ArrayList;"
    ],
    "javascript":[]
}

LANGUAGE_TO_TIMEOUT = {
    "python": 60,
    "cpp": 60,
    "javascript": 60,
    "java": 60,
}

# Java sometimes fails with more workers; For javascript it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 4,
}

LANGUAGES = ["python", "cpp", "javascript", "java"]
PERTUBATION = ["no_change","rename","code_stmt_exchange","code_expression_exchange","insert","code_style"]

#// mbpp_generate_cpp_robust_rename,mbpp_generate_cpp_robust_code_stmt_exchange,mbpp_generate_cpp_robust_code_expression_exchange,mbpp_generate_cpp_robust_code_style
ALL_INSTRUCT_TASKS = "mbpp_generate_python_robust_no_change_instruct,mbpp_generate_python_robust_rename_instruct,\
mbpp_generate_python_robust_code_stmt_exchange_instruct,mbpp_generate_python_robust_code_expression_exchange_instruct,\
mbpp_generate_python_robust_insert_instruct,mbpp_generate_python_robust_code_style_instruct,\
mbpp_generate_cpp_robust_no_change_instruct,mbpp_generate_cpp_robust_rename_instruct,mbpp_generate_cpp_robust_code_stmt_exchange_instruct,\
mbpp_generate_cpp_robust_code_expression_exchange_instruct,\
mbpp_generate_cpp_robust_insert_instruct,mbpp_generate_cpp_robust_code_style_instruct,\
mbpp_generate_javascript_robust_no_change_instruct,mbpp_generate_javascript_robust_rename_instruct,mbpp_generate_javascript_robust_code_stmt_exchange_instruct,\
mbpp_generate_javascript_robust_code_expression_exchange_instruct,mbpp_generate_javascript_robust_insert_instruct,\
mbpp_generate_javascript_robust_code_style_instruct,\
mbpp_generate_java_robust_no_change_instruct,mbpp_generate_java_robust_rename_instruct,mbpp_generate_java_robust_code_stmt_exchange_instruct,\
mbpp_generate_java_robust_code_expression_exchange_instruct,mbpp_generate_java_robust_insert_instruct,mbpp_generate_java_robust_code_style_instruct"

ALL_COMPLETION_TASKS = "mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename,mbpp_generate_python_robust_code_stmt_exchange,\
mbpp_generate_python_robust_code_expression_exchange,mbpp_generate_python_robust_insert,\
mbpp_generate_python_robust_code_style,\
mbpp_generate_cpp_robust_no_change,mbpp_generate_cpp_robust_rename,mbpp_generate_cpp_robust_code_stmt_exchange,\
mbpp_generate_cpp_robust_code_expression_exchange,mbpp_generate_cpp_robust_insert,mbpp_generate_cpp_robust_code_style,\
mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_code_stmt_exchange,\
mbpp_generate_javascript_robust_code_expression_exchange,mbpp_generate_javascript_robust_insert,\
mbpp_generate_javascript_robust_code_style,\
mbpp_generate_java_robust_no_change,mbpp_generate_java_robust_rename,\
mbpp_generate_java_robust_code_stmt_exchange,mbpp_generate_java_robust_code_expression_exchange,\
mbpp_generate_java_robust_insert,mbpp_generate_java_robust_code_style"

def create_all_tasks():
    ret = {}
    for type in ['robust']:
        for language in LANGUAGES:
            for pertubation in PERTUBATION:
                ret[f"mbpp_generate_{language}_{type}_{pertubation}_instruct"] = create_task(type, language,pertubation,model_type="causal_chat")
    
    for type in ['robust']:
        for language in LANGUAGES:
            for pertubation in PERTUBATION:
                ret[f"mbpp_generate_{language}_{type}_{pertubation}"] = create_task(type, language,pertubation,model_type="causal_base")

    return ret


def create_task(type,language,pertubation,model_type):
    class MBPPgenerate(GeneralMbppGenerate):
        def __init__(self):
            super().__init__(type,language,pertubation,model_type=model_type)

    return MBPPgenerate



class GeneralMbppGenerate(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(self, type, language,pertubation,k=[1, 5, 10, 100], model_series="", model_type="causal_chat"):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )
        self.type = type
        self.language = language
        self.perturbation = pertubation
        self.k = k
        self.model_series = model_series
        self.model_type = model_type
        self.code_col_name = "code_str_deleted" if self.model_type == "causal_chat" else "code_str_generate"
        self.DATASET_PATH = f"dataset/mbpp_{language}_tested.json"
        self.p = PerturbationPipeline()
        self.p.set_seed(42)
        self.p.init_pretrained_model()
        self.p.preprocess_code('',self.language)
        self.perturbate = {'rename':self.p.rename_perturbation,'code_stmt_exchange':self.p.code_stmt_perturbtion,\
                           'code_expression_exchange':self.p.code_expression_perturbtion,'insert':self.p.insert_perturbation,\
                            'code_style':self.p.code_style_perturbtion,'no_change':self.p.no_change_perturbation}
        
        self.patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
        self.task_ids = []
        self.perturbation_types = []

    def get_dataset(self):
        self.dataset = json.load(open(self.DATASET_PATH, "r"))
        #self.dataset = self.dataset[:10] # debug
        self.count = 0
        self.real_dataset = []
        indexs = []
        perturbated_code = []
        perturbated_test = []
        perturbation_types = []
        for i in range(len(self.dataset)):
            perturbations_one_time = self.perturbate[self.perturbation]() #针对于某个样本的扰动复制
            while perturbations_one_time:
                real_pertubertion = random.choice(perturbations_one_time) #从中随机选择一个扰动
                real_pertubertion_copy = real_pertubertion
                perturbation_type = real_pertubertion[1]
                real_pertubertion = real_pertubertion[0]
                code_before = self.dataset[i][self.code_col_name]
                is_perturbated = False
                if 'func' in perturbation_type: #只有重命名函数名会将code和test接在一起进行扰动，因为test中函数名也会跟着变
                    code = self.dataset[i][self.code_col_name] + "\n" + self.dataset[i]['test']
                    code = real_pertubertion(code).strip() #应用扰动
                    test = re.search(self.patterns[self.language],code).group(0) #测试用例
                    code = re.sub(self.patterns[self.language],'',code) #没有测试用例的代码
                else:
                    code = self.dataset[i][self.code_col_name]
                    code = real_pertubertion(code).strip() #应用扰动
                    test = self.dataset[i]['test']
                is_perturbated = not(code.strip() == code_before.strip()) #如果不相等，说明扰动成功
                if is_perturbated or perturbation_type == "no_change": #如果扰动成功，将扰动结果加入，并结束扰动
                    indexs.append(i)
                    perturbation_types.append(perturbation_type)
                    perturbated_code.append(code)
                    perturbated_test.append(test)
                    break
                else: #如果扰动失败，删除此扰动方式，继续随机选择一个扰动，直到所有扰动都被选择
                    perturbations_one_time.remove(real_pertubertion_copy)
        
        assert len(perturbated_code) == len(perturbated_test) == len(indexs)
        task_ids = []
        for i,v in enumerate(perturbated_code):
            data = {}
            data['code_str'] = perturbated_code[i]
            data['test'] = perturbated_test[i]
            data[f'{self.language}_prompt'] = self.dataset[indexs[i]][f'{self.language}_prompt']
            data['index'] = indexs[i]
            task_ids.append(indexs[i])
            self.real_dataset.append(copy.deepcopy(data))
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        self.dataset = copy.deepcopy(self.real_dataset)
        self.task_ids = copy.deepcopy(task_ids)
        self.perturbation_types = copy.deepcopy(perturbation_types)
        p = f"mbpp_generate_{self.language}_robust_{self.perturbation}_instruct.json" if self.model_type == "causal_chat" \
            else f"mbpp_generate_{self.language}_robust_{self.perturbation}.json"
        json.dump(self.task_ids,open(os.path.join("task_ids",p),"w"),indent=4)
        return self.dataset

    def get_prompt(self, doc):
        if self.model_type == "causal_base":
            return doc['code_str']
        code = doc['code_str']
        order = doc[f'{self.language}_prompt']
        x = ""
        if self.language == "cpp":
            x = "5. Do not generate a main function, as I have my own main function available."
        elif self.language == "java":
            x = "5. Do not modify class \"Solution\" as a public class."
        elif self.language == "python":
            x = "5. Mind indent in python code."
        elif self.language == "javascript":
            x = "5. Do not generate \"console.log\" statement, do not use \"require\" to import package."

        prompt = r"""
This is a code generation task. Please help me write the code. The programming language for the code is {}. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.
The prompt for the code is: {}
The code content is:
-----------------------------
{}
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
{}

Format:
```{}
Complete code (including all the content of the code I provided and the code you generated)
```
""".format(self.language,order,code,x,self.language)
        return prompt   

    def get_reference(self, doc):
        return doc['test']
            
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """

        prompt = self.get_prompt(self.dataset[idx])
        #print(generation)
        if self.model_type == "causal_chat":
            generation = generation[len(prompt):]
        print("original_generation:---------------")
        print(generation)
        generation = extract_code(generation, self.language)
        print("extract code:-------------")
        print(generation)
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        def minDistance( word1: str, word2: str) -> int:#编辑距离函数
            n1 = len(word1)
            n2 = len(word2)

            dp = [[inf for _ in range(n2+1)] for _ in range(n1+1)] #i表示word1索引，j表示word2索引
            dp[0][0] = 0 #空字符串匹配不需要操作
            for i in range(1,n1+1):
                dp[i][0] = dp[i-1][0] + 1 #word1[:i]和空字符串匹配需要删除字符
                
            for j in range(1,n2+1):
                dp[0][j] = dp[0][j-1] + 1 #空字符串匹配和word2[:j]匹配需要插入字符

            for i in range(1,n1+1):
                for j in range(1,n2+1):
                    if word1[i-1] == word2[j-1]:#i-1和j-1是当前匹配字符的索引
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) + 1
            return dp[-1][-1]
        
        code_metric = load("bigcode_eval/tasks/custom_metrics/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.language]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.language]
        import_helper = "\n".join(IMPORT_HELPER[self.language])
        length = len(generations)
        generations = [
           [(import_helper + "\n" + g).strip() for g in gen] for gen in generations
           ]

        for i,v in enumerate(generations):
            gen_func_name = self.p.get_function_names(v[0])[1]
            ori_func_name = self.p.get_invoke_func_names(references[i])[1]
            #print(references[i])
            #print(gen_func_name,ori_func_name)
            if gen_func_name and ori_func_name:
                if len(gen_func_name) > 1:
                    n_min = gen_func_name[0]
                    d_min = minDistance(n_min,ori_func_name[0]) #只能通过编辑距离去判断到底是哪个函数名
                    for n in gen_func_name[1:]:
                        d = minDistance(n,ori_func_name[0])
                        if d < d_min:
                            n_min = n
                else:
                    n_min = gen_func_name[0]
                references[i] = self.p.rename_function_name(references[i],ori_func_name[0],n_min)

        metrics, cases = code_metric.compute(
            references=references,
            predictions=generations,
            language= self.language,
            timeout=timeout,
            num_workers=num_workers,
            k = self.k
        )

        cases_return = {}

        for i in range(len(self.task_ids)):
            inner = {}
            for k,v in cases[i][0][1].items():
                inner[k] = v
            inner['task_id'] = self.task_ids[i]
            inner['perturbation_type'] = self.perturbation_types[i]
            cases_return[i] = copy.deepcopy(inner)
            #print(cases_return)
        stat = {}
        stat["name"] = {"name": "full_level_robustness", "split": "test"}
        stat["count"] = length//2
        stat['generations'] = generations
        stat["variance"] = 0.0
        stat["stddev"] = 0.0

        stats = [[stat, f"robust_mbpp_perturbation:mbpp_generate_{self.language}_{self.type}_{self.perturbation}-generation-generation,"]]

        return metrics, cases_return, stats
