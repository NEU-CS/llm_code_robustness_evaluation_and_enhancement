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
    "javascript":[""]
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "javascript": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For javascript it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

CPPHEAD = """
#undef NDEBUG
#include<assert.h>
bool issame(int a,vector<int>b){
    if (b.size() > 1) return false;
    if (a!=b[0]) return false;
    return true;
};
bool issame(vector<string> a,vector<string>b){
    if (a.size()!=b.size()) return false;
    for (int i=0;i<a.size();i++)
    {
    if (a[i]!=b[i]) return false;
    }
    return true;
}
bool issame(vector<int> a,vector<int>b){
    if (a.size()!=b.size()) return false;
    for (int i=0;i<a.size();i++)
    {
    if (a[i]!=b[i]) return false;
    }
    return true;
};
"""
LANGUAGES = ["python", "cpp", "javascript", "java", "go", "rust"]

cpp_template ="""
"""

javascript_template = """
格式：
```javascript
const testHasCloseElements = () => {
新生成的第一条测试样例
新生成的第二条测试样例
新生成的第三条测试样例
新生成的第四条测试样例
新生成的第五条测试样例
}
"""

java_template = """
```java
public class Main{
    public static void main(String[] args) {
        Solution solution = new Solution();
            新生成的第一条测试样例
            新生成的第二条测试样例
            新生成的第三条测试样例
            新生成的第四条测试样例
            新生成的第五条测试样例
    }
}
```
"""

python_template = """
格式：
```{self.language}
def check(candidate):
    新生成的第一条测试样例
    新生成的第二条测试样例
    新生成的第三条测试样例
    新生成的第四条测试样例
    新生成的第五条测试样例
check({entry_point})
```"""

def create_all_tasks():
    return {f"humaneval_implementation-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalIm(GeneralHumanEvalIm):
        def __init__(self):
            super().__init__(language)

    return HumanEvalIm

class GeneralHumanEvalIm(Task):
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
        self.DATASET_PATH = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/dataSet/implementation/humaneval_{language}_im.json"
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
        code = doc[f"{self.language}_code_str"]
        test = doc[f"{self.language}_test"]
        test_cases5 = doc[f"{self.language}_test_cases5"]
        entry_point = doc["entry_point"]
        prompt = f"""
我有一段代码，以及几条测试样例，但是测试样例太少了，我希望你帮我多生成几条不同的测试用例。
代码内容：
```{self.language}
{code}
```

已知输出结果的测试样例
```{self.language}
{test}
```

未知输出结果的测试样例
```{self.language}
{test_cases5}
```

要求：
1、不需要解释原因，也不需要给出其他文字信息；
2、请按照已知输出结果的测样样例的格式，给出对应的输出结果，一行为一条测试样例；
3、请完整给出每一条未知输出结果的测试样例的输入，以及"=="之后的输出内容；
4、请给出所有未知输出结果的测试样例的输出结果；
5、请按照以下格式给出你的回答。
"""
        if self.language == 'python':
            prompt = prompt + '\n' + f"""
格式：
```{self.language}
def check(candidate):
    新生成的第一条测试样例
    新生成的第二条测试样例
    新生成的第三条测试样例
    新生成的第四条测试样例
    新生成的第五条测试样例
check({entry_point})
```
"""
        if self.language == 'javascript':
            prompt = prompt + '\n' + f"""
格式：
```{self.language}
const test{entry_point}""" + """ = () => {
新生成的第一条测试样例
新生成的第二条测试样例
新生成的第三条测试样例
新生成的第四条测试样例
新生成的第五条测试样例
}
```
""" + f"""test{entry_point}()"""
        if self.language == 'java':
            prompt = prompt + '\n' + """
格式：
```java
新生成的第一条测试样例
新生成的第二条测试样例
新生成的第三条测试样例
新生成的第四条测试样例
新生成的第五条测试样例     
```
"""

        if self.language == 'cpp' :
            prompt = prompt + '\n' + """
格式：
#undef NDEBUG
#include<assert.h>
int main(){
新生成的第一条测试样例
新生成的第二条测试样例
新生成的第三条测试样例
新生成的第四条测试样例
新生成的第五条测试样例
}
"""
        return prompt   

    def get_reference(self, doc):
        reference = doc[f"{self.language}_code_str"]
        import_helper = "\n".join(IMPORT_HELPER[self.language])
        if self.language == 'cpp':
            return import_helper + CPPHEAD + reference
        return import_helper + '\n' + reference
        
    def replace_between_correct(self, code, gen):
        lines = code.split('\n')
        
        result_lines = []
        inside_correct_block = False
        start_index = -1

        for index, line in enumerate(lines):
            if 'correct' in line:
                if not inside_correct_block:
                    # 发现第一个 correct
                    inside_correct_block = True
                    result_lines.append(line)
                else:
                    # 发现第二个 correct
                    inside_correct_block = False
                    result_lines.append(gen)  # 在正确位置插入 'hello'
                    result_lines.append(");")
                    result_lines.append(line)     # 保留第二个 correct 标签所在行
            elif not inside_correct_block:
                result_lines.append(line)

        return '\n'.join(result_lines)

    # def replace_middle_lines(self, input_string, replacement_string):
    # # 将字符串按行分割
    #     lines = input_string.split('\n')
        
    #     # 构建新的行列表，首行和末行保持不变，中间行替换
    #     new_lines = [lines[0]]  # 首行
    #     new_lines.extend([replacement_string] * (len(lines) - 2))  # 中间行
    #     new_lines.append(lines[-1])  # 末行
        
    #     # 将新行列表重新拼接成字符串
    #     output_string = '\n'.join(new_lines)
    #     return output_string

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        pattern = re.compile(r"```{}(.*?)```".format(self.language), re.DOTALL)
        prompt = self.get_prompt(self.dataset[idx])
        ori_code = re.findall(pattern, prompt)
        ori_code = ori_code[1]
        generation = generation[len(prompt):]
        match = pattern.search(generation) 
        if self.language == 'java':
            if match:
                u = match.group(1).strip()
                java_t = u.split('\n')
                for i in range(len(java_t) - 1):
                    java_t[i] = java_t[i] + ','
                output_java = '\n'.join(java_t)
                return self.replace_between_correct(ori_code, output_java) 
            else:
                return generation
        else:
            if match:
                print(match.group(1).strip())
                return match.group(1).strip() 
            else:
                return generation
        



    def process_results(self, generations, reference):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        code_metric = load("./bigcode_eval/tasks/custom_metrics/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.language]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.language]
        import_helper = "\n".join(IMPORT_HELPER[self.language])
        # if self.language != 'java':
        #     generations = [
        #     [(import_helper + "\n" + g).strip() for g in gen] for gen in generations]


        metrics, cases = code_metric.compute(
            references=[item for sublist in generations for item in sublist],
            predictions=[[ref] for ref in reference],
            language= self.language,
            timeout=timeout,
            num_workers=num_workers,
        )

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(reference)
        sum_ = metrics["pass@1"] * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = metrics["pass@1"]
        stat["max"] = metrics["pass@1"]
        stat["mean"] = metrics["pass@1"]
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        stats = [[stat, "humaneval_implemetation_zero_shot-generation-generation:"]]

        return metrics, cases, stats
