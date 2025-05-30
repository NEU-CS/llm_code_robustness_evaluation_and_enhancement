"""Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
https://openreview.net/forum?id=1qvx610Cu7

The HumanEval+ dataset is created by the EvalPlus framework which extends the original HumanEval dataset
by adding more automatically generated test cases to each problem.

Homepage: https://github.com/evalplus/evalplus
"""

from warnings import warn

from bigcode_eval.tasks.humaneval import *

_CITATION = """
@inproceedings{evalplus,
  title = {Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation},
  author = {Liu, Jiawei and Xia, Chunqiu Steven and Wang, Yuyao and Zhang, Lingming},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year = {2023},
  url = {https://openreview.net/forum?id=1qvx610Cu7},
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        "humanevalplus": create_task(True),
        "humanevalplus-unstripped": create_task(False),
    }


def create_task(strip_prompt):
    class HumanEvalPlus(GeneralHumanEvalPlus):
        def __init__(self, model_series, model_type, stop_words, **kwargs):
            super().__init__(strip_prompt, model_series=model_series, model_type=model_type, 
                             stop_words=stop_words, **kwargs)

    return HumanEvalPlus


class GeneralHumanEvalPlus(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "evalplus/humanevalplus"
    DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/humanevalplus"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=20.0,
                 model_series="", model_type="causal_base", stop_words=[]):
        if timeout < 20.0:
            warn(
                "It is suggested to have a longer timeout as HumanEval+ has lots of tests. "
                f"The current timeout is {timeout}s while the suggested timeout is 20s."
            )
        super().__init__(strip_prompt, k, num_workers, timeout, model_series, model_type, stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        python_imports = "\n".join(IMPORT_HELPER)
        generations = [
            [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
        ]

        metrics, cases = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
            task="humanevalplus",
        )

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(references)
        sum_ = metrics["pass@1"] * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = metrics["pass@1"]
        stat["max"] = metrics["pass@1"]
        stat["mean"] = metrics["pass@1"]
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        stats = [[stat, "humanevalplus_zero_shot-generation-generation:"]]

        return metrics, cases, stats
