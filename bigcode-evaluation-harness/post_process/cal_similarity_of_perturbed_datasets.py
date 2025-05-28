import sys
sys.path.append('../adv_samples_gen')
from CodeBLEU.calc_code_bleu import compute_metrics
from transformers import AutoTokenizer
import json
import heapq
import copy
import random
import re
import os
sys.path.append("../perturbation_pipeline")
from pipeline import PerturbationPipeline



def get_dataset(dataset,model_type,language,perturbation):
        code_col_name = "code_str_deleted" if model_type == "instruct" else "code_str_generate"
        p = PerturbationPipeline()
        p.set_seed(42)
        p.init_pretrained_model()
        p.preprocess_code('',language)
        perturbate = {'rename':p.rename_perturbation,'code_stmt_exchange':p.code_stmt_perturbtion,\
                                'code_expression_exchange':p.code_expression_perturbtion,'insert':p.insert_perturbation,\
                                    'code_style':p.code_style_perturbtion,'no_change':p.no_change_perturbation}
        patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                        'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                            'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                                'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
        real_dataset = []
        indexs = []
        perturbated_code = []
        perturbated_test = []
        perturbation_types = []
        for i in range(len(dataset)):
            perturbations_one_time = perturbate[perturbation]() #针对于某个样本的扰动复制
            while perturbations_one_time:
                real_pertubertion = random.choice(perturbations_one_time) #从中随机选择一个扰动
                real_pertubertion_copy = real_pertubertion
                perturbation_type = real_pertubertion[1]
                real_pertubertion = real_pertubertion[0]
                code_before = dataset[i][code_col_name]
                is_perturbated = False
                if 'func' in perturbation_type: #只有重命名函数名会将code和test接在一起进行扰动，因为test中函数名也会跟着变
                    code = dataset[i][code_col_name] + "\n" + dataset[i]['test']
                    code = real_pertubertion(code).strip() #应用扰动
                    test = re.search(patterns[language],code).group(0) #测试用例
                    code = re.sub(patterns[language],'',code) #没有测试用例的代码
                else:
                    code = dataset[i][code_col_name]
                    code = real_pertubertion(code).strip() #应用扰动
                    test = dataset[i]['test']
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
            data[f'{language}_prompt'] = dataset[indexs[i]][f'{language}_prompt']
            data['index'] = indexs[i]
            task_ids.append(indexs[i])
            real_dataset.append(copy.deepcopy(data))
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = copy.deepcopy(real_dataset)
        task_ids = copy.deepcopy(task_ids)
        perturbation_types = copy.deepcopy(perturbation_types)
        return dataset



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct")
    LANGAUGES = ['cpp','python','java','javascript']
    for perturbation in ["rename","code_stmt_exchange","code_expression_exchange","code_style","insert"]:
        for lang in LANGAUGES:
            dataset = json.load(open(f'dataset/mbpp_{lang}_tested.json'))
            perturbated_dataset = get_dataset(dataset,"instruct",lang,perturbation)
            codebleu = [0 for _ in range(len(perturbated_dataset))]
            for i in range(len(perturbated_dataset)):
                index = perturbated_dataset[i]['index']
                original_code = dataset[index]['code_str_deleted']
                perturbed_code = perturbated_dataset[i]['code_str']
                cb = compute_metrics((tokenizer(original_code,return_tensors="pt").input_ids,tokenizer(perturbed_code, return_tensors="pt").input_ids),tokenizer,lang)
                codebleu[i] = cb['CodeBLEU']
            print(lang,perturbation,min(codebleu),max(codebleu),sum(codebleu)/len(codebleu))
            json.dump(codebleu,open(f"dataset/original_perturbed_similarity/{lang}_{perturbation}.json","w"),indent=4)
    