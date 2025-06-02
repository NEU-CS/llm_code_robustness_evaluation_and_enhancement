import os
import copy
import json
import inspect
import warnings

from transformers import AutoTokenizer

from bigcode_eval import tasks
from bigcode_eval.api_utils import _make_instruction_prompt
from bigcode_eval.api_client import HttpAPIServer
from bigcode_eval.filewriter import write_save_generations

_WARNING = """
"""


class APIEvaluator:
    def __init__(self, model_name, args, global_args):
        self.model_name = model_name
        self.args = args
        self.global_args = global_args
        # setup server
        self.server = HttpAPIServer(model_name=model_name, global_args=global_args)

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution
        

    def generate_text(self, task_name, global_args):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # n_tasks = self.args.limit if self.args.limit else len(dataset)
        n_tasks = min(self.args.limit, len(dataset)) if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        prompts = []
        generations = []
        gen_kwargs = {
            "temperature": global_args["temperature"],
            "top_k": global_args["top_k"],
            "top_p": global_args["top_p"],
            "do_sample": global_args["do_sample"],
            "max_length": global_args["max_new_tokens"],
        }

        tokenizer = None
        try:
            if self.args.tokenizer_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_path,
                    revision=self.args.revision,
                    trust_remote_code=self.args.trust_remote_code,
                    use_auth_token=self.args.use_auth_token,
                    truncation_side="left",
                    padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
                )

        except:
            # 尝试加载tokenizer 能加载就加载 不能先算了...
            pass

        for sample in range(self.args.limit_start, self.args.limit_start+n_tasks):
            prompts_, generations_ = [], []
            for _ in range(self.args.n_samples):
                prompt_contents = task.get_prompt(dataset[sample])
                prompt = None
                if isinstance(prompt_contents, str):
                    # Normal code completion mode
                    prompt = self.args.prefix + prompt_contents + self.args.suffix

                elif isinstance(prompt_contents, dict):
                    if set(prompt_contents.keys()) == {"prefix", "middle", "suffix", "prompt"}:
                        # Infilling mode
                        prompt = prompt_contents["prompt"]
                        # prompt = _make_infill_prompt(
                        #     **prompt_contents, preprefix=self.args.prefix
                        # )
                    elif set(prompt_contents.keys()) == {"instruction", "context"}:
                        # Instruction-tuning mode
                        prompt = _make_instruction_prompt(
                            **prompt_contents, prefix=self.args.prefix, suffix=self.args.suffix
                        )
                else:
                    raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")

                # Tokenizer截断
                if tokenizer is not None and self.args.max_model_length > 0:
                    encoded_prompt = tokenizer.encode(
                        prompt, add_special_tokens=False
                    )
                    actual_len = len(encoded_prompt)
                    max_length = self.args.max_model_length - self.args.max_new_tokens
                    if actual_len > max_length:
                        encoded_prompt = encoded_prompt[-(max_length - 1) :]
                        prompt = tokenizer.decode(
                            encoded_prompt, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        print(f"prompt超长，实际长度: {actual_len}, 目标长度: {max_length}, 截断后文本: {prompt}", flush=True)

                prompts_.append(prompt)
                raw_request = {}
                raw_request["prompt"] = prompt
                completions = self.server.serve_request(raw_request, global_args, **gen_kwargs)
                generation = completions["completions"][0]["text"]
                if (not self.args.model_type == "causal_base") and self.args.suffix is not None and len(self.args.suffix) > 0:
                    prompt = prompt[: -len(self.args.suffix)]
                prompt = prompt[len(self.args.prefix) :]
                generation = task.postprocess_generation(prompt+generation, sample+self.args.limit_start)
                generations_.append(generation)
            prompts.append(prompts_)
            generations.append(generations_)

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return prompts, generations, references

    def evaluate(self, task_name, global_args):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        prompts, generations, references = self.generate_text(task_name, global_args)

        if not self.args.load_generations_path:
            if self.args.save_generations:
                save_generations_dir = f"result/{self.args.model_name}/{self.args.save_generations_path}"
                save_generations_path = f"{save_generations_dir}/{task_name}"
                write_save_generations(prompts, generations, references, save_generations_path)

        # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        metrics, cases, stats = task.process_results(generations, references)
        return metrics, cases, stats
