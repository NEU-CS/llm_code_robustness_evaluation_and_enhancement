"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import extract_python_code

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "mbpp"
    DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/mbpp"

    def __init__(self, model_series, model_type, stop_words):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )
        self.model_series = model_series
        self.model_type = model_type
        self.use_instruction = False
        self.instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the python code for this function.\n"
        if self.model_type == "causal_chat":
            self.use_instruction = True
            self.stop_words = ["if __name__", "\nprint", "\nclass", "\nassert"]
        self.stop_words += stop_words

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        # test_example = doc["test_list"][0]
        # prompt = f'"""\n{description}\n{test_example}\n"""\n'
        test_example = "\n".join(doc["test_list"])
        prompt = f'"""\n{description}\n{test_example}\n"""'
        if self.use_instruction:
            return self.instruction + prompt
        else:
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        description = self.dataset["test"]["text"][idx]
        test_example = "\n".join(self.dataset["test"]["test_list"][idx])
        context = f'"""\n{description}\n{test_example}\n"""'
        generation = generation[len(prompt) :]

        if not self.use_instruction:
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = context + generation
        else:
            generation = extract_python_code(generation)
            generation = self._stop_at_stop_token(generation, self.stop_words).lstrip()
            generation = "\n" + generation
            generation = context + generation

        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        metrics, cases = compute_code_eval(
            references=references,
            predictions=generations,
            task="mbpp",
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
        stats = [[stat, "mbpp_zero_shot-generation-generation:"]]

        return metrics, cases, stats
