import inspect
import json
import os
import warnings

from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations
from bigcode_eval.filewriter import write_save_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.task = None
        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        self.task = tasks.get_task(task_name, self.args)
        dataset = self.task.get_dataset()
        # if args.limit is None, use all samples
        # n_tasks = self.args.limit if self.args.limit else len(dataset)
        n_tasks = min(self.args.limit, len(dataset)) if self.args.limit else len(dataset)
        references = [self.task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(self.task.get_reference).parameters:
                solutions = [[self.task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        prompts, generations = parallel_generations(
            self.task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return prompts, generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)
        prompts, generations, references = self.generate_text(task_name)
        task = self.task
        if self.accelerator.is_main_process:
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

        return None, None, None
