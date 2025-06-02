# BigCode Evaluation Harness

This is our model evaluation framework, which supports:
- Custom tasks
- Custom datasets  
- Adaptation to different models

**Custom Tasks Location**:  
`bigcode-evaluation-harness/bigcode_eval/tasks/private_algorithmic`

**Framework Details**:  
[Original BigCode-Evaluation-Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)  
(A specialized framework for evaluating language models' coding capabilities)

---

## Task Implementations

| File Path | Purpose |
|-----------|---------|
| `bigcode_eval/tasks/private_algorithmic/robust_mbpp_generate_instruct.py` | Robustness evaluation |
| `bigcode_eval/tasks/private_algorithmic/robust_mbpp_generate_preprocess.py` | Code standardization |
| `bigcode_eval/tasks/private_algorithmic/robust_mbpp_generate_prompt.py` | In-context learning |

> All prompts are auto-generated and defined in respective code files (`get_prompt()` function)

---

## Prompt Examples

<div class="prompt-example">

### Robustness Evaluation
```python
This is a code generation task. Please help me write the code. The programming language for the code is python. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.
The prompt for the code is: Write a python function to remove first and last occurrence of a given character from the string.
The code content is:
-----------------------------
def remove_Occ(a,c):
    for j in range(len(a)): 
        if (a[j] == c): 
            a = a[0 : j] + a[j + 1:] 
            break
            #begin to write code
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
5. Mind indent in python code.

Format:
```python
Complete code (including all the content of the code I provided and the code you generated)
```


</div>

<div class="prompt-example">

### In-Context Learning
```python
This is a code generation task. Please help me write the code. The programming language for the code is python. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.

Here are examples of this task:

Code content:
def find_Divisor(x,y):
    if (x==y): 
    #begin to write code

Complete code:
def find_Divisor(x,y):  
    if (x==y): 
        return y 
    return 2

Code content:
def count_occurance(s):
  count=0
  for i in range(len(s)):
  #begin to write code

Complete code:
def count_occurance(s):
  count=0
  for i in range(len(s)):
    if (s[i]== 's' and s[i+1]=='t' and s[i+2]== 'd'):
      count = count + 1
  return count


Code content:
def test_distinct(data):
  if len(data) == len(set(data)):
    return True
    #begin to write code

Complete code:
def test_distinct(data):
  if len(data) == len(set(data)):
    return True
  else:
    return False;

Code content:
def ascii_value_string(str1):
  for i in range(len(str1)):
  #begin to write code

Complete code:
def ascii_value_string(str1):
  for i in range(len(str1)):
   return ord(str1[i])

Code content:
def all_unique(test_list):
    if len(test_list) > len(set(test_list)):
    #begin to write code

Complete code:
def all_unique(test_list):
    if len(test_list) > len(set(test_list)):
        return False
    return True

Please help complete the following code.
The prompt for the code is: Write a python function to remove first and last occurrence of a given character from the string.
The code content is:
-----------------------------
def remove_Occ(a,c):
    for j in range(len(a)): 
        if (a[j] == c): 
            a = a[0 : j] + a[j + 1:] 
            break
            #begin to write code
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
5. Mind indent in python code.

Format:
```python
Complete code (including all the content of the code I provided and the code you generated)
```
</div>

<div class="prompt-example">

### Code Standardization
```text
This is a code generation task. Please help me write the code. The programming language for the code is python. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.
Please help complete the following code.
The prompt for the code is: Write a python function to remove first and last occurrence of a given character from the string.
The code content is:
-----------------------------
def remove_Occ(a,c):
    for j in range(len(a)): 
        if (a[j] == c): 
            a = a[0 : j] + a[j + 1:] 
            break
            #begin to write code
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
5. Mind indent in python code.

Format:
```python
Complete code (including all the content of the code I provided and the code you generated)
```