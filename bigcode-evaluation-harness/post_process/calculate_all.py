

from evaluate import load
import os
import argparse
import json
import sys
from math import inf
import re
sys.path.append("../perturbation_pipeline")
from pipeline import PerturbationPipeline

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

'''
根据结果文件中的cases判断apply perturbation部分的pass@1是多少,然后计算pass-drop@1
'''
if __name__ == '__main__':
    MODELS = ['codellama-7b-instruct','deepseek-v2-lite-chat','llama3.1-8b-instruct',\
        'llama3.2-1b-instruct','qwen2.5-7b-instruct','qwen2.5-coder-0.5b-instruct',\
            'qwen2.5-coder-1.5b-instruct','qwen2.5-coder-3b-instruct','qwen2.5-coder-7b-instruct',\
            'codellama-7b-base','deepseek-v2-lite-base',\
                'llama3.1-8b-base','llama3.2-1b-base','Qwen2.5-7b-base','Qwen2.5-coder-0.5b-base',\
                    'Qwen2.5-coder-1.5b-base','Qwen2.5-coder-3b-base','Qwen2.5-coder-7b-base']
    #codeQwen1.5-7B-base
    PERTURBATION = ['code_style',"insert","rename","code_stmt_exchange","code_expression_exchange"]
    LANGUAGES = ['python','cpp','java','javascript']
    parser = argparse.ArgumentParser()
    for model_name in ['qwen2.5-coder-0.5b-instruct-prompt']:
        for perturbation in ['code_stmt_exchange']:
            for language in LANGUAGES:
                for prompt_type in ["1random_prompt","3random_prompt","5random_prompt"]:
                    model_type = "causal_base" if model_name.endswith("base") else "causal_chat"
                    print(f"calculate model:{model_name} perturbation:{perturbation} language:{language}")
                    prefix_path = "/data1/ljc/code/llm_robustness_eval_and_enhance/bigcode-evaluation-harness/result/"
                    perturbation_whole_name = ""
                    if model_type == "causal_base":
                        perturbation_whole_name = f"mbpp_generate_{language}_robust_{perturbation}_{prompt_type}"
                        original_cases = json.load(open(os.path.join(prefix_path,model_name,"evaluation_results",f"mbpp_generate_{language}_robust_no_change_{prompt_type}","evaluation_cases.json"),"r"))
                        
                    else:
                        perturbation_whole_name = f"mbpp_generate_{language}_robust_{perturbation}_instruct_{prompt_type}"
                        original_cases = json.load(open(os.path.join(prefix_path,model_name,"evaluation_results",f"mbpp_generate_{language}_robust_no_change_instruct_{prompt_type}","evaluation_cases.json"),"r"))

                    print(perturbation_whole_name)
                    perturbed_results = json.load(open(os.path.join(prefix_path,model_name,"evaluation_results",perturbation_whole_name,"evaluation_results.json"),"r"))
                    perturbed_cases = json.load(open(os.path.join(prefix_path,model_name,"evaluation_results",perturbation_whole_name,"evaluation_cases.json"),"r"))

                    

                    original_pass_at_1 = inf
                    perturbed_pass_at_1 = perturbed_results[perturbation_whole_name]['perturbated_pass@1']
                    match_cases = 0
                    match_passes = 0
                    neg2pos = 0
                    pos2neg = 0

                    for k,v in perturbed_cases.items():
                        index = str(v['task_id'])
                        match_cases += 1
                        if original_cases[index]['passed'] == True:
                            assert original_cases[index]['task_id'] == int(index)
                            match_passes += 1
                            if v['passed'] == False:
                                pos2neg += 1
                        
                        if v['passed'] == True:
                            if original_cases[index]['passed'] == False:
                                neg2pos += 1

                    original_pass_at_1 = match_passes/match_cases
                    pass_drop_at_1 = round((original_pass_at_1-perturbed_pass_at_1)/(original_pass_at_1+0.00001),3)

                        
                    results = {perturbation_whole_name:{"origin_pass@1":original_pass_at_1,"perturbated_pass@1":perturbed_pass_at_1,"pass-drop@1":pass_drop_at_1,"neg2pos":neg2pos,"pos2neg":pos2neg,"count":match_cases}}
                    json.dump(results,open(os.path.join(prefix_path,model_name,"evaluation_results",perturbation_whole_name,"evaluation_results.json"),"w"),indent=4)






                

