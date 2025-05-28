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
python evaluate_pass_at.py --language=java\
  --model_name=qwen2.5-coder-3b-instruct\
  --perturbation=no_change\
  --model_type=causal_chat\
'''
ps = ["rename","code_style","insert","code_stmt_exchanging","code_expression_exchanging"]
if __name__ == '__main__':
    base_dir = "result"
    model = "qwen2.5-coder-1.5b-instruct-adv-train"
    next_dir = os.path.join(base_dir,model,"generations")
    for every in os.listdir(next_dir):
        if "cpp" in every:
            x = json.load(open(os.path.join(next_dir,every,"generations.json")))
            store = []
            count = 0
            for y in x:
                y = y[0]
                #1. 大括号匹配；
                #2. 行拆分符号替换
                lines = y.split('\n')
                real_lines = []
                index = 0
                while index < len(lines):
                    oneline = lines[index]
                    if oneline.endswith("\\"):
                        oneline = oneline[:-1]
                        index += 1
                        real_lines.append(oneline + lines[index])
                    else:
                        real_lines.append(oneline)
                    index += 1
                
                def match(st):
                    #大括号匹配，如果最后一个大括号无法匹配则删除
                    #print(st)
                    stack = []
                    record = []
                    for i in range(len(st)):
                        #print(stack)
                        if st[i] == r'{':
                            stack.append(st[i])
                        elif st[i] == r'}':

                            if stack and stack[-1] == r'{':
                                stack.pop(-1)
                            else:
                                record.append(i)
                                return st[:i] + st[i+1]

                    return st
                
                y = match(y)
                store.append([y])
            print(os.path.join(next_dir,every,"generations2.json"))
            json.dump(store,open(os.path.join(next_dir,every,"generations2.json"),"w"))

    




                

