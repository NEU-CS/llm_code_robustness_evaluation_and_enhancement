import os
import fnmatch
import json
import time
import warnings

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig
)

from bigcode_eval.api_client import HttpAPIServer
from bigcode_eval.api_evaluator import APIEvaluator
from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.filewriter import write_save_generations, write_metric_output, write_helm_output
from bigcode_eval.tasks import ALL_TASKS


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model_name",
        default="deepseek-coder-v2-lite-base",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--model_path",
        default="/data1/model/deepseek/deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        help="local model path",
    )
    parser.add_argument(
        "--model_type",
        default="causal_base",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum length of generated sequence (generation)",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=0,
        help="Maximum length of generated sequence (generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default="auto",
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    ### API params
    parser.add_argument(
        "--api",
        type=str,
        default=None,
        help="api",
    )
    parser.add_argument(
        "--model_series",
        type=str,
        default=None,
        help="model_series, e.g. llama",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="model_version, e.g. 7b",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer path, the tokenizer path of the model",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="benchmark",
    )
    parser.add_argument(
        "--flash_atten",
        type=str,
        default=None,
        help="flash-attention",
    )
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    # humanevalpack_explain task requires generating results before evaluation.
    humanevalpack_explain_tasks = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            if matching.startswith("humanevalpack_explain"):
                humanevalpack_explain_tasks.add(f'humanevalpack_explain-{matching.split("-")[-1]}')
            else:
                task_names.add(matching)
    task_names = list(task_names)
    humanevalpack_explain_tasks = list(humanevalpack_explain_tasks)
    task_names.sort()
    humanevalpack_explain_tasks.sort()
    task_names += [task.replace("humanevalpack_explain", "humanevalpack_explaindescribe") for task in
                   humanevalpack_explain_tasks]
    task_names += [task.replace("humanevalpack_explain", "humanevalpack_explainsynthesize") for task in
                   humanevalpack_explain_tasks]
    return task_names


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    args = parse_args()
    if args.n_samples == 1:
        args.temperature = 0.0
        args.top_k = 1
        args.top_p = 0.0
        args.do_sample = False
    if args.model_series == "deepseek-r1":
        pass
        #args.temperature = 0.6
    
    args.prefix = args.prefix.replace("\\n", "\n")
    args.suffix = args.suffix.replace("\\n", "\n")
    stop_words = args.eos.split("||")
    stop_words = list(set([stop_word for stop_word in stop_words if len(stop_word) > 0]))
    args.stop_words = stop_words
    global_args = vars(args)
    print("global_args:", global_args)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    use_api = HttpAPIServer.use_api_client(model_name=args.model_name, global_args=global_args)

    res_dir = "result"
    if not use_api:
        accelerator = Accelerator()
        if accelerator.is_main_process:
            print(f"Selected Tasks: {task_names}")

        results = {}
        if args.load_generations_path:
            # here we don't generate code but only evaluate previously computed generations
            if accelerator.is_main_process:
                print("evaluation only mode")
            evaluator = Evaluator(accelerator, None, None, args)
            for task in task_names:
                print(f"Now evaluation task:{task}")
                results[task] = evaluator.evaluate(task)
        else:
            # here we generate code and save it (evaluation is optional but True by default)
            dict_precisions = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            if args.precision not in dict_precisions:
                raise ValueError(
                    f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
                )

            model_kwargs = {
                "revision": args.revision,
                "trust_remote_code": args.trust_remote_code,
                "use_auth_token": args.use_auth_token,
            }
            if args.load_in_8bit:
                print("Loading model in 8bit")
                model_kwargs["load_in_8bit"] = args.load_in_8bit
                model_kwargs["device_map"] = {"": accelerator.process_index}
            elif args.load_in_4bit:
                print("Loading model in 4bit")
                model_kwargs["load_in_4bit"] = args.load_in_4bit
                model_kwargs["device_map"] = {"": accelerator.process_index}
            else:
                print(f"Loading model in {args.precision}")
                model_kwargs["torch_dtype"] = dict_precisions[args.precision]

                if args.max_memory_per_gpu:
                    if args.max_memory_per_gpu != "auto":
                        model_kwargs["max_memory"] = get_gpus_max_memory(
                            args.max_memory_per_gpu, accelerator.num_processes
                        )
                        model_kwargs["offload_folder"] = "offload"
                    else:
                        model_kwargs["device_map"] = "auto"
                        print("Loading model in auto mode")
            if args.flash_atten:
                model_kwargs["attn_implementation"] = args.flash_atten #添加flash-attention加速推理

            input_model = args.model_name if args.model_path is None else args.model_path
            if args.model_type.startswith("causal"):
                model = AutoModelForCausalLM.from_pretrained(
                    input_model,
                    **model_kwargs,
                )
                if args.model_series == "deepseek-v2":
                    #model.generation_config = GenerationConfig.from_pretrained(args.model_name)
                    model.generation_config.pad_token_id = model.generation_config.eos_token_id
            elif args.model_type == "seq2seq":
                warnings.warn(
                    "Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models."
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    input_model,
                    **model_kwargs,
                )
            else:
                raise ValueError(
                    f"Non valid model_type {args.model_type}, choose from: causal, seq2seq"
                )

            if args.peft_model:
                from peft import PeftModel  # dynamic import to avoid dependency on peft

                model = PeftModel.from_pretrained(model, args.peft_model)
                print("Loaded PEFT model. Merging...")
                model.merge_and_unload()
                print("Merge complete.")

            tokenizer = AutoTokenizer.from_pretrained(
                input_model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                truncation_side="left",
                padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
            )

            if not tokenizer.eos_token:
                if tokenizer.bos_token:
                    tokenizer.eos_token = tokenizer.bos_token
                    print("bos_token used as eos_token")
                else:
                    raise ValueError("No eos_token or bos_token found")
            if not tokenizer.pad_token:
                try:
                    tokenizer.pad_token = tokenizer.eos_token
                # Some models like CodeGeeX2 have pad_token as a read-only property
                except AttributeError:
                    print("Not setting pad_token to eos_token")
                    pass
            # WIZARD_LLAMA_MODELS = [
            #     "WizardLM/WizardCoder-Python-34B-V1.0",
            #     "WizardLM/WizardCoder-34B-V1.0",
            #     "WizardLM/WizardCoder-Python-13B-V1.0"
            # ]
            # # if input_model in WIZARD_LLAMA_MODELS:
            #     tokenizer.bos_token = "<s>"
            #     tokenizer.bos_token_id = 1
            #     print("Changing bos_token to <s>")

            evaluator = Evaluator(accelerator, model, tokenizer, args)
            for task in task_names:
                print(f"Now evaluation task:{task}")
                if args.generation_only or task.startswith("humanevalpack_explaindescribe"):
                    if accelerator.is_main_process:
                        print("generation mode only")
                    prompts, generations, references = evaluator.generate_text(task)
                    if accelerator.is_main_process:
                        save_generations_dir = f"{res_dir}/{args.model_name}/{args.save_generations_path}"
                        save_generations_path = f"{save_generations_dir}/{task}"
                        write_save_generations(prompts, generations, references, save_generations_path)
                else:
                    metrics, cases, stats = evaluator.evaluate(task)
                    if accelerator.is_main_process:
                        results[task] = metrics
                        metric_output_dir = f"{res_dir}/{args.model_name}/{args.metric_output_path}"
                        metric_output_path = f"{metric_output_dir}/{task}"
                        write_metric_output(metrics, cases, metric_output_path, task)
                        write_helm_output(stats, metric_output_dir, task, args.model_name, args.benchmark)

        # Save all args to config
        results["config"] = vars(args)
        if not args.generation_only:
            if accelerator.is_main_process:
                dumped = json.dumps(results, indent=2)
                print(dumped)

    else:
        print("This way is using api !!!")
        print(f"Selected Tasks: {task_names}")

        results = {}
        evaluator = APIEvaluator(args.model_name, args, global_args)
        for task in task_names:
            if args.generation_only or task.startswith("humanevalpack_explaindescribe"):
                prompts, generations, references = evaluator.generate_text(task, global_args)
                save_generations_dir = f"{res_dir}/{args.model_name}/{args.save_generations_path}"
                save_generations_path = f"{save_generations_dir}/{task}"
                write_save_generations(prompts, generations, references, save_generations_path)
            else:
                metrics, cases, stats = evaluator.evaluate(task, global_args)
                results[task] = metrics
                metric_output_dir = f"{res_dir}/{args.model_name}/{args.metric_output_path}"
                metric_output_path = f"{metric_output_dir}/{task}"
                write_metric_output(metrics, cases, metric_output_path, task)
                write_helm_output(stats, metric_output_dir, task, args.model_name, args.benchmark)

        # Save all args to config
        results["config"] = vars(args)
        if not args.generation_only:
            dumped = json.dumps(results, indent=2)
            print(dumped)

    # Avoid compressing files while still writing files
    time.sleep(60)


if __name__ == "__main__":
    main()
