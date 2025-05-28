# coding=utf-8
import sys
import os
sys.path.append('./python_parser')
sys.path.append('../bigcode-evaluation-harness')
from bigcode_eval.utils import extract_code
sys.path.append("../perturbation_pipeline")
from pipeline import PerturbationPipeline

import json
import logging
import argparse
import warnings
import torch
import time
from utils_adv import Recorder,set_seed,IMPORT_HELPER,LANGUAGE_TO_NUM_WORKERS,LANGUAGE_TO_TIMEOUT,LANGUAGES
from attacker_gen import Attacker
from transformers import pipeline,AutoTokenizer
from datasets import load_dataset
from torch.functional import F
from evaluate import load
from CodeBLEU.calc_code_bleu import compute_metrics
from tqdm import tqdm
from datasets import Dataset
import re
from math import inf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\
logger = logging.getLogger(__name__)
os.environ['HF_ALLOW_CODE_EVAL']= '1'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'



'''
1. 数据集改成mbpp(四种语言都要有);
2. 评价指标是pass@k(要把bigcode里面的test搬过来);
3. 代码转换方式是pipeline中的五种
'''


class myclassifier():
    def __init__(self,classifier):
        self.classifier = classifier
        self.querytimes = 0

    def predict(self,code):
        if type(code['code']) == list:
            self.querytimes += len(code['code'])
        elif type(code['code']) == str:
            self.querytimes += 1 
        return self.classifier(code)
    
    def query(self):
        return self.querytimes
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--language", default="python",type=str,
                        help="programming language now testing")
    parser.add_argument("--model_type",default="base",type=str,
                        help="instruct or base model")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--perturbation_type", type=str,
                        help="perturbation_type name in [rename,code_stmt_exchange,code_expression_exchange,insert,code_style,no_change]")
    parser.add_argument("--use_sa", action='store_true',
                        help="Whether to simulated annealing-Attack.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="beam size of beam search")
    parser.add_argument('--iter_nums', type=int, default=10,
                        help="")
    parser.add_argument('--transfrom_iters', type=int, default=1,
                        help="")
    parser.add_argument('--p', type=float, default=0.8,
                        help="")
    parser.add_argument('--accptance', type=float, default=0.005,
                        help="")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")



    
    
    args = parser.parse_args()
    args.eval_data_file = f"/data1/ljc/code/llm_robustness_eval_and_enhance/intern_files/dataset/generate/mbpp_{args.language}_tested.json"
    # Set seed
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator = pipeline("text-generation",model = args.model_name_or_path,device = args.device)
    p = PerturbationPipeline()
    p.set_seed(42)
    p.preprocess_code('',args.language)
    #pass@k code_metric evalaution
    code_metric = load("../bigcode-evaluation-harness/bigcode_eval/tasks/custom_metrics/code_eval_octopack")
    patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
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
    
    def Causalpipeline():
        #注意如果是替换function_name，一定要把测试用例中的function_name也替换掉
        def get_output_and_eval_res(code,doc):
            #在这里进行inference，把code作为输入给到模型，模型输出返回，并提取代码传入metirc计算中
            if type(code) == list:
                order = doc[f'{args.language}_prompt']
                references = [doc["test"] for _ in code] #提取测试用例
                x = ""
                if args.language == "cpp":
                    x = "5. Do not generate a main function, as I have my own main function available."
                elif args.language == "java":
                    x = "5. Do not modify class \"Solution\" as a public class."
                elif args.language == "python":
                    x = "5. Mind indent in python code."
                elif args.language == "javascript":
                    x = "5. Do not generate \"console.log\" statement, do not use \"require\" to import package."

                prompts_instruct = [r"""
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
""".format(args.language,order,c,x,args.language) for c in code]
                
                prompts_complete = code
                prompts = prompts_instruct if args.model_type == "instruct" else prompts_complete
                dataset = Dataset.from_dict({"text": prompts})
                # 定义一个函数来处理每个样本
                st = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                diff_x = [0 for _ in prompts]
                def process_batch(batch):
                    # 使用 pipeline 进行推理
                    nonlocal diff_x
                    if args.model_type == "instruct":
                        messages = [[{"role": "system", "content": st,\
                                  "role": "user", "content": prompt}] for prompt in batch["text"]]
                        formatted_input = tokenizer.apply_chat_template(messages,tokenize=False)
                        diff_x = [len(formatted_input[i]) - len(prompts[i]) for i in range(len(prompts))]
                        results = generator(formatted_input,max_new_tokens=256, batch_size=args.eval_batch_size)
                        return {"results": results}
                    else:
                        results = generator(batch["text"],max_new_tokens=256, batch_size=args.eval_batch_size)
                        return {"results": results}
            
                #model predict是并行化的
                import_helper = "\n".join(IMPORT_HELPER[args.language])
                output = dataset.map(process_batch, batched=True, batch_size=args.eval_batch_size)
                generations = []
                for i,o in enumerate(output['results']):
                    if args.model_type == "instruct":
                        o = o[0]['generated_text'][len(prompts[i])+diff_x[i]:]
                    else:
                        o = o[0]['generated_text']
                    generations.append(extract_code(o,args.language))
                print(o)
                ret = generations
                generations = [
                    [(import_helper + "\n" + gen).strip()] for gen in generations
                    ]
                timeout = LANGUAGE_TO_TIMEOUT[args.language]
                num_workers = LANGUAGE_TO_NUM_WORKERS[args.language]

                metrics_ret = []
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

                for i,v in enumerate(generations):
                    metrics_origin, cases_origin = code_metric.compute(
                    references=[references[i]],
                    predictions=[generations[i]],
                    language= args.language,
                    timeout=timeout,
                    num_workers=num_workers,
                )   
                    pass_at_1 = metrics_origin['pass@1']
                    #源代码进行eval时，直接使用pass@1作为指标
                    #扰动后代码进行eval时，如果pass@1为0，直接使用pass@1；否则，pass@1和codebleu同时作为评价指标，权重各为0.5
                    if not doc['flag']:
                        codebleu = compute_metrics((tokenizer(generations[i],return_tensors="pt").input_ids,tokenizer(doc['code_str'], return_tensors="pt").input_ids),tokenizer,args.language)
                        codebleu = codebleu['CodeBLEU']
                        if pass_at_1 == 0:
                            metrics_ret.append(pass_at_1)
                        else:
                            metrics_ret.append(codebleu * 0.5 + pass_at_1 * 0.5)
                    else:
                        metrics_ret.append(pass_at_1)
                    
                #metrics evaluation是串行化的
            else:
                raise Exception
            assert len(ret) == len(code)

            return ret,metrics_ret
        
        def infer(example):
            ret = []
            if type(example['code']) == list:
                generations,res = get_output_and_eval_res(example['code'],example['doc'])
                for i,v in enumerate(generations):
                    ret.append({'code':v,'score':res[i]}) 
            else:
                raise Exception
            return ret
        
        return infer
    classifier = Causalpipeline()
    classifier = myclassifier(classifier)

    ## Load Dataset
    eval_dataset = load_dataset("json", data_files = args.eval_data_file)
    eval_dataset = eval_dataset['train']
    success_attack = 0
    total_cnt = 0
    #print(args.model_name_or_path.split('/'))
    csv_store_path = f"results/{args.model_name_or_path.split('/')[-1]}/{args.language}/{args.perturbation_type}.csv"
    recoder = Recorder(csv_store_path)
    attacker = Attacker(classifier,args.language,args.iter_nums,args.transfrom_iters,args.perturbation_type,args.use_sa,args.p,args.accptance,args.beam_size,args.model_type)
    start_time = time.time()
    query_times = 0
    for index, example in tqdm(enumerate(eval_dataset)):
        example_start_time = time.time()
        original_code,prog_length,ground_truth,orig_prediction,adv_code,adv_truth,adv_prediction,is_success\
                ,adv_perturbation_type,orig_prob,current_prob,attack_path = attacker.greedy_attack(example)
    
        example_end_time = (time.time()-example_start_time)/60
        
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time()-start_time)/60, 2), "min")
        
        print("Query times in this attack: ", classifier.query() - query_times)
        print("All Query times: ", classifier.query())

        recoder.write(index, original_code,prog_length,ground_truth,orig_prediction,adv_code,adv_truth,adv_prediction,\
            is_success,classifier.query() - query_times,example_end_time,f"{args.perturbation_type}:{adv_perturbation_type}",\
                orig_prob,current_prob,attack_path)
        query_times = classifier.query()
        
        if is_success >= -1 :
            total_cnt += 1
        if is_success == 1:
            success_attack += 1
            
        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
    
    
if __name__ == '__main__':
    main()
