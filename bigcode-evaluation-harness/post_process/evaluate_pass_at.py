from evaluate import load
import os
import argparse
import json
import sys
from math import inf
sys.path.append("../../perturbation_pipeline")
from pipeline import PerturbationPipeline

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
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
        "#include<cassert>"
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

def minDistance( word1: str, word2: str) -> int:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--model_name", default="", type=str,
                        help="")
    parser.add_argument("--perturbation", default="", type=str,
                        help="")
    parser.add_argument("--model_type", default="", type=str,
                        help="")
    parser.add_argument("--prompt_type",default="",type=str)
    parser.add_argument("--language",default="", type=str)
    args = parser.parse_args()
    
    code_metric = load("bigcode_eval/tasks/custom_metrics/code_eval_octopack")
    references = []
    generations = []
    prompts = []
    prefix_path = "../result"
    result_path = os.path.join(prefix_path,args.model_name)
    generation_path = os.path.join(result_path,"generations",f"mbpp_generate_{args.language}_robust_{args.perturbation}_instruct{args.prompt_type}") if args.model_type == "causal_chat" else\
    os.path.join(result_path,"generations",f"mbpp_generate_{args.language}_robust_{args.perturbation}")

    with open(os.path.join(generation_path,"generations2.json"),"r") as f:
        generations = json.loads(f.read())
    with open(os.path.join(generation_path,"references.json"),"r") as f:
        references = json.loads(f.read())
    with open(os.path.join(generation_path,"prompts.json"),"r") as f:
        prompts = json.loads(f.read())

    print(f"evaluate pass@1 of model:{args.model_name} perturbation:{args.perturbation} language:{args.language}")

    timeout = LANGUAGE_TO_TIMEOUT[args.language]
    num_workers = LANGUAGE_TO_NUM_WORKERS[args.language]
    import_helper = "\n".join(IMPORT_HELPER[args.language])
    generations = [
           [(import_helper + "\n" + g).strip() for g in gen] for gen in generations
           ]
    p = PerturbationPipeline()
    p.preprocess_code("",args.language)
    #print(references)
    for i,v in enumerate(generations):
        gen_func_name = p.get_function_names(v[0])[1]
        ori_func_name = p.get_invoke_func_names(references[i])[1]
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
            references[i] = p.rename_function_name(references[i],ori_func_name[0],n_min)
        #break
  
    
    metrics_origin, cases_origin = code_metric.compute(
                references=references,
                predictions=generations,
                language= args.language,
                timeout=timeout,
                num_workers=num_workers,
            )
    evaluation_path = os.path.join(result_path,"evaluation_results",f"mbpp_generate_{args.language}_robust_{args.perturbation}_instruct{args.prompt_type}") if args.model_type == "causal_chat" else\
    os.path.join(result_path,"evaluation_results",f"mbpp_generate_{args.language}_robust_{args.perturbation}{args.prompt_type}")
    metrics_final = {f"mbpp_generate_{args.language}_robust_{args.perturbation}_instruct{args.prompt_type}":{"pass@1":metrics_origin["pass@1"]}}
    json.dump(metrics_origin,open(os.path.join(evaluation_path,"evaluation_results.json"),"w"),indent=4)
    json.dump(cases_origin,open(os.path.join(evaluation_path,"evaluation_cases.json"),"w"),indent=4)
    
    

