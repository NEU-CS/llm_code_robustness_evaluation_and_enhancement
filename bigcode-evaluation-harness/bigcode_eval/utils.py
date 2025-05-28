import math
import re
import torch
import warnings

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import IterableDataset
from typing import Any, Dict, List


INFILL_MODE = False
INSTRUCTION_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length=0,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        suffix="",
        has_encoder=False,
        model_series="",
        model_type="",
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix
        self.suffix = suffix
        self.has_encoder = has_encoder
        self.model_series = model_series
        self.model_type = model_type
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        prompts_encoder = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start+self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents + self.suffix
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "middle", "suffix", "prompt"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = prompt_contents["prompt"]
                    # prompt = self._make_infill_prompt(
                    #     **prompt_contents, preprefix=self.prefix
                    # )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix, suffix=self.suffix
                    )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)
            if self.has_encoder:
                prompt_encoder = self.task.get_prompt_encoder(self.dataset[sample])
                if isinstance(prompt_encoder, str):
                    prompt_encoder = self.prefix + prompt_encoder + self.suffix
                prompts_encoder.append(prompt_encoder)
        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        if self.model_type == "causal_chat":
            try:
                if self.model_series == "qwen2.5":
                    messages = [[{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",\
                                  "role": "user", "content": prompt}] for prompt in prompts]
                elif self.model_series == "qwen1.5":
                    messages = [[{"role": "system", "content": "You are a helpful assistant.",\
                                  "role": "user", "content": prompt}] for prompt in prompts]
                                
                else:
                    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
                outputs = self.tokenizer.apply_chat_template(messages,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length if self.max_length > 0 else None,
                    add_generation_prompt=True,
                    return_dict=True,
                )
            except Exception as e:
                print(e)
                outputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length if self.max_length > 0 else None,
                    return_token_type_ids=return_token_type_ids,
                )
        else:
            outputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length if self.max_length > 0 else None,
                return_token_type_ids=return_token_type_ids,
            )
            if self.has_encoder:
                outputs_encoder = self.tokenizer(
                    prompts_encoder,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length if self.max_length > 0 else None,
                    return_token_type_ids=return_token_type_ids,
                )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                if self.has_encoder:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "ids_encoder": outputs_encoder.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                        "input_len_encoder": outputs_encoder.attention_mask[sample].sum(),
                    }
                else:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                    }

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix="", suffix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context + suffix
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="",
    suffix="",
    customize_prompt=False,
    model_series="",
    model_type="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    print(gen_kwargs)
    system_prompt = ""
    if model_type == "causal_chat":
        try:
            messages = [{"role": "user", "content": ""}]
            system_prompt_inputs = tokenizer.apply_chat_template(messages,
                truncation=True,
                return_tensors="pt",
                # max_length=self.max_length,
                add_generation_prompt=True,
            )[0]
            if system_prompt_inputs[0] == tokenizer.bos_token_id:
                system_prompt_inputs = system_prompt_inputs[1:]
            if system_prompt_inputs[-1] == tokenizer.eos_token_id:
                system_prompt_inputs = system_prompt_inputs[:-1]
            system_prompt = tokenizer.decode(
                    system_prompt_inputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
        except Exception as e:
            print(e)
            pass

    pro_token_dict = defaultdict(list) # dict of list of prompted tokens
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                max_len = batch["input_len"].max().item()
                if "ids_encoder" in batch:
                    max_len += 1 # Add 1 for decoder_start_token_id
                gen_kwargs["stopping_criteria"][0].start_length = max_len
            if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
                idx = 1 if task.stop_words else 0
                gen_kwargs["stopping_criteria"][idx].input_length = batch["input_len"].max().item()
            
            inputs = batch["ids"][:, : batch["input_len"]]
            if "ids_encoder" in batch:
                if is_wrapped:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
                else:
                    generated_tokens = model.generate(
                        decoder_input_ids=inputs,
                        input_ids=batch["ids_encoder"][:, : batch["input_len_encoder"]],
                        num_return_sequences=batch_size,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
            else:
                if is_wrapped:
                    # 8bit and 4bit models are wrapped in accelerator
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )
                else:
                    generated_tokens = model.generate(
                        input_ids=inputs,
                        num_return_sequences=batch_size,
                        **gen_kwargs,
                    )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, prompted_tokens in zip(generated_tasks, inputs):
                pro_token_dict[sample].append(prompted_tokens)
            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

    code_pros = [[] for _ in range(n_tasks)]
    code_gens = [[] for _ in range(n_tasks)]

    for sample, prompted_tokens in pro_token_dict.items():
        generated_tokens = gen_token_dict[sample]
        for p, g in zip(prompted_tokens, generated_tokens):
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                if p[0] == tokenizer.bos_token_id:
                    p = p[1:]
                if p[-1] == tokenizer.eos_token_id:
                    p = p[:-1]
                if g[0] == tokenizer.bos_token_id:
                    g = g[1:]
                if g[-1] == tokenizer.eos_token_id:
                    g = g[:-1]
                # Treat eos token as a regular stop word not removing it from the output
                # If it's removed it may have the effect of removing it in the middle of a
                # longer generation in case a batch size > 1 is used, which will result in
                # a wrong generation as it won't be used for splitting lateron 
                pro_code = tokenizer.decode(
                    p, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                gen_code = tokenizer.decode(
                    g, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                # if INFILL_MODE:
                    # pro_code = _parse_infill(pro_code, tokenizer)
                # if INSTRUCTION_MODE:
                #     pro_code = _parse_instruction(pro_code, instruction_tokens)
            else:
                pro_code = tokenizer.decode(
                    p, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_code = tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            code_pros[sample].append(pro_code)

            if not INFILL_MODE:
                # gen_code = gen_code[len(prefix) :]
                gen_code = gen_code[len(pro_code) :]
                if model_series == "codellama" and model_type == "causal_chat":
                    gen_code = gen_code[2 :]
                if (not customize_prompt) and suffix is not None and len(suffix) > 0:
                    pro_code = pro_code[: -len(suffix)]
                pro_code = pro_code[len(prefix) + len(system_prompt) :]
                gen_code = pro_code + gen_code
            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)

    # for sample, generated_tokens in gen_token_dict.items():
    #     for i, s in enumerate(generated_tokens):
    #         if INFILL_MODE or tokenizer.eos_token in task.stop_words:
    #             if s[0] == tokenizer.bos_token_id:
    #                 s = s[1:]
    #             if s[-1] == tokenizer.eos_token_id:
    #                 s = s[:-1]
    #             # Treat eos token as a regular stop word not removing it from the output
    #             # If it's removed it may have the effect of removing it in the middle of a
    #             # longer generation in case a batch size > 1 is used, which will result in
    #             # a wrong generation as it won't be used for splitting lateron
    #             gen_code = tokenizer.decode(
    #                 s, skip_special_tokens=False, clean_up_tokenization_spaces=False
    #             )
    #             # if INFILL_MODE:
    #             #     gen_code = _parse_infill(gen_code, tokenizer)
    #             # if INSTRUCTION_MODE:
    #             #     gen_code = _parse_instruction(gen_code, instruction_tokens)
    #         else:
    #             gen_code = tokenizer.decode(
    #                 s, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #         print("INFILL_MODE: ", INFILL_MODE)
    #         if not INFILL_MODE:
    #             print("gen_code prefix: ", gen_code)
    #             # gen_code = gen_code[len(prefix):]
    #             pro_code = code_pros[sample][i]
    #             gen_code = gen_code[len(pro_code):]
    #             pro_code = pro_code[len(prefix): -len(suffix)]
    #             gen_code = pro_code + gen_code
    #             print("gen_code: ", gen_code)
    #         if postprocess:
    #             code_gens[sample].append(
    #                 task.postprocess_generation(gen_code, int(sample) + limit_start)
    #             )
    #         else:
    #             warnings.warn(
    #                 "model output is not postprocessed, this might lower evaluation scores"
    #             )
    #             code_gens[sample].append(gen_code)

    return code_pros, code_gens

patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}

def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON3]", "```python").replace("[/PYTHON3]", "```")
    generation = generation.replace("[PYTHON]", "```python").replace("[/PYTHON]", "```")
    generation = generation.replace("```python3", "```python")
    generation = generation.replace("`python", "```python")

    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    if len(matches) >= 1: 
        code_block = matches[0]
        return code_block

    sep_index = generation.find("```")
    if sep_index != -1:
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(generation)
        if len(matches) >= 1: 
            code_block = matches[0]
            return code_block
        elif generation[sep_index + len("```") : sep_index + len("```python")] == "python":
            generation = generation[sep_index + len("```python") :]
        else:
            generation = generation[sep_index + len("```") :]
    return generation


def extract_cpp_code(generation: str):
    generation = generation.replace("[CPP]", "```cpp").replace("[/CPP]", "```")
    generation = generation.replace("[C++]", "```cpp").replace("[/C++]", "```")
    generation = generation.replace("[C]", "```c").replace("[/C]", "```")
    generation = generation.replace("```c++", "```cpp")

    
    if re.search(patterns["cpp"],generation):
        generation = re.sub(patterns["cpp"],"",generation) #如果生成了主函数，移除主函数
    pattern = re.compile(r"```cpp\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    if len(matches) >= 1: 
        code_block = matches[0]
        return code_block

    pattern = re.compile(r"```c\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    if len(matches) >= 1: 
        code_block = matches[0]
        return code_block

    sep_index = generation.find("```")
    if sep_index != -1:
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(generation)
        if len(matches) >= 1: 
            code_block = matches[0]
            return code_block
        elif generation[sep_index + len("```") : sep_index + len("```cpp")] == "cpp":
            generation = generation[sep_index + len("```cpp") :]
        elif generation[sep_index + len("```") : sep_index + len("```c")] == "c":
            generation = generation[sep_index + len("```c") :]
        else:
            generation = generation[sep_index + len("```") :]
    
    return generation


def extract_java_code(generation: str):
    generation = generation.replace("[JAVA]", "```java").replace("[/JAVA]", "```")
    generation = generation.replace("`java", "```java")

    if re.search(patterns["java"],generation):
        generation = re.sub(patterns["java"],"",generation) #如果生成了主函数，移除主函数
    
    '''
    pattern = re.compile(r"public\s+static\s+void\s+main.+\{.+\}",re.DOTALL)
    if re.search(pattern,generation):
        generation = re.sub(pattern,"",generation) #去除public static main函数
    '''
    
    pattern = re.compile(r"(?P<public_name>public)?\s+class\s+(?P<class_name>[_0-9a-zA-Z]+)\s*\{")
    generation = re.sub(pattern,r"class Solution {",generation)#不管模型生成的类名是什么，全都替换为Solution，并且将public移除
    pattern = re.compile(r"```java\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    pattern_end = re.compile(r"class Solution \{.*",re.DOTALL) #只要class Solution{}之内的内容，其他都不要
    if len(matches) >= 1: 
        code_block = matches[0]
        generation_end = pattern_end.findall(code_block)
        if generation_end:
            generation_end[0] = re.sub("```","",generation_end[0])
            return generation_end[0]
        else:
            code_block = re.sub("```","",code_block)
            return code_block

    sep_index = generation.find("```")
    if sep_index != -1:
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(generation)
        if len(matches) >= 1: 
            code_block = matches[0]
            code_block = re.sub("```","",code_block)
            return code_block
        elif generation[sep_index + len("```") : sep_index + len("```java")] == "java":
            generation = generation[sep_index + len("```java") :]
        else:
            generation = generation[sep_index + len("```") :]

    generation_end = pattern_end.findall(generation)
    if generation_end:
        generation_end[0] = re.sub("```","",generation_end[0])
        return generation_end[0]
    else:
        generation = re.sub("```","",generation)
        return generation


def extract_sh_code(generation: str):
    generation = generation.replace("[BASH]", "```bash").replace("[/BASH]", "```")

    pattern = re.compile(r"```bash\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    if len(matches) >= 1: 
        code_block = matches[0]
        return code_block

    sep_index = generation.find("```")
    if sep_index != -1:
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(generation)
        if len(matches) >= 1: 
            code_block = matches[0]
            return code_block
        elif generation[sep_index + len("```") : sep_index + len("```bash")] == "bash":
            generation = generation[sep_index + len("```bash") :]
        else:
            generation = generation[sep_index + len("```") :]
    return generation


def extract_js_code(generation: str):
    generation = generation.replace("[JAVASCRIPT]", "```javascript").replace("[/JAVASCRIPT]", "```")
    generation = generation.replace("`javascript", "```javascript")
    pattern = re.compile(r"console.log.+") #删掉console.log，这会影响test case判断
    if pattern.search(generation):
        generation = re.sub(pattern,"",generation)

    pattern = re.compile(r"```javascript\n(.*?)```", re.DOTALL)
    matches = pattern.findall(generation)
    if len(matches) >= 1: 
        code_block = matches[0]
        return code_block

    sep_index = generation.find("```")
    if sep_index != -1:
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(generation)
        if len(matches) >= 1: 
            code_block = matches[0]
            return code_block
        elif generation[sep_index + len("```") : sep_index + len("```javascript")] == "javascript":
            generation = generation[sep_index + len("```javascript") :]
        else:
            generation = generation[sep_index + len("```") :]
    return generation


def extract_code(generation: str, language: str):
    if language in ["python", "py"]:
        return extract_python_code(generation)
    elif language in ["cpp"]:
        return extract_cpp_code(generation)
    elif language in ["java"]:
        return extract_java_code(generation)
    elif language in ["sh"]:
        return extract_sh_code(generation)
    elif language in ["js", "javascript"]:
        return extract_js_code(generation)
    return generation


def remove_after_return(code: str):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0:start_match]
        end_last_match = end_match
    return code