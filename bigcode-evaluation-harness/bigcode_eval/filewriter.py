import os
import json
from typing import Any, Dict, List


def write_save_generations(prompts: List[List[str]], generations: List[List[str]], references: List[str], 
                           save_generations_path: str):
    os.makedirs(save_generations_path, exist_ok=True)
    with open(f"{save_generations_path}/prompts.json", "w", encoding="UTF-8") as fp:
        json.dump(prompts, fp, ensure_ascii=False)
        print(f"prompts were saved at {save_generations_path}/prompts.json")

    with open(f"{save_generations_path}/generations.json", "w", encoding="UTF-8") as fp:
        json.dump(generations, fp, ensure_ascii=False)
        print(f"generations were saved at {save_generations_path}/generations.json")

    with open(f"{save_generations_path}/references.json", "w", encoding="UTF-8") as fp:
        json.dump(references, fp, ensure_ascii=False)
        print(f"references were saved at {save_generations_path}/references.json")


def write_metric_output(metrics: Dict[str, Any], cases: Dict[str, Any], 
                        metric_output_path: str, task: str):
    os.makedirs(metric_output_path, exist_ok=True)
    with open(f"{metric_output_path}/evaluation_results.json", "w", encoding="UTF-8") as f:
        dumped = json.dumps({task: metrics}, indent=2)
        print(dumped)
        f.write(dumped)

    with open(f"{metric_output_path}/evaluation_cases.json", "w", encoding="UTF-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def write_helm_scenario_state(prompts: List[List[str]], generations: List[List[str]], references: List[str], 
                              helm_output_path: str, model_name: str):

    adapter_spec = {
        "model": model_name,
    }
    request_states = []
    ids = [x for x in range(len(prompts))]
    for prompt, generation, reference, idx in zip(prompts, generations, references, ids):
        sub_ids = [x for x in range(len(prompt))]
        for p, g, sub_id in zip(prompt, generation, sub_ids):
            request_states.append({
                "instance": {
                    "input": {
                        "text": p
                    },
                    "references": [
                        {
                            "output": {
                                "text": reference if reference is not None else ""
                            },
                            "tags": [
                                "correct"
                            ]
                        }],
                    "split": "test",
                    "id": f"id{idx}-{sub_id}",
                    "access_control": "PUBLIC"
                },
                "train_trial_index": 0,
                "request": {
                    "model": model_name,
                    "embedding": False,
                    "prompt": p,
                    "prompt_prefix": "",
                    "prompt_suffix": "",
                    "temperature": 0.0,
                    "num_completions": 1,
                    "top_k_per_token": 1,
                    "max_tokens": 0,
                    "stop_sequences": [
                        "</s>"
                    ],
                    "echo_prompt": False,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "agent_labels": []
                },
                "result": {
                    "success": True,
                    "embedding": [],
                    "completions": [
                        {
                            "text": g,
                            "logprob": 0,
                            "tokens": [],
                            "correct_logprob": 0.0
                        }
                    ],
                    "cached": False,
                    "request_time": 0.0,
                    "request_datetime": 0,
                    "agent_labels": []
                },
                "num_train_instances": 0,
                "prompt_truncated": False,
                "num_conditioning_tokens": 0,
                "num_rounds": 1,
                "multi_turn_requests": [],
                "multi_turn_results": []
            })

    scenario_state = {"adapter_spec": adapter_spec, "request_states": request_states}
    with open(f"{helm_output_path}/scenario_state.json", "w", encoding="UTF-8") as f:
        json.dump(scenario_state, f, indent=2, ensure_ascii=False)


def write_helm_per_instance_stats(prompts: List[List[str]], generations: List[List[str]], references: List[str], 
                                  cases: Dict[str, Any], helm_output_path: str, model_name: str):
    per_instance_stats = []
    ids = [x for x in range(len(prompts))]
    for prompt, generation, reference, idx in zip(prompts, generations, references, ids):
        sub_ids = [x for x in range(len(prompt))]
        for p, g, sub_id in zip(prompt, generation, sub_ids):
            try:
                case_stats = 1 if cases[str(idx)][sub_id][-1]["passed"] else 0
            except:
                case_stats = 0
            per_instance_stats.append({
                "instance_id": f"id{idx}-{sub_id}",
                "train_trial_index": 0,
                "stats": [
                    {
                        "name": {
                            "name": "quasi_prefix_exact_match",
                            "split": "test",
                        },
                        "count": 1,
                        "sum": case_stats,
                        "sum_squared": case_stats,
                        "min": case_stats,
                        "max": case_stats,
                        "mean": case_stats,
                        "variance": 0,
                        "stddev": 0
                    },
                ]
            })
    with open(f"{helm_output_path}/per_instance_stats.json", "w", encoding="UTF-8") as f:
        json.dump(per_instance_stats, f, indent=2, ensure_ascii=False)


def write_helm_stats(stat: Dict[str, Any], helm_output_path: str, model_name: str, dataset: str):
    os.makedirs(helm_output_path, exist_ok=True)
    with open(f"{helm_output_path}/stats.json", "w", encoding="UTF-8") as f:
        json.dump([stat], f, indent=2, ensure_ascii=False)
    f.close()

    run_config = {
        "dataset": dataset[:-1],
        "model": model_name,
        "name": f'{dataset}model={model_name.replace("/", "_")}',
    }
    dataset = run_config["dataset"].split("-")[0]
    benchmark = dataset.split(":")[0]
    run_spec = {
        "name": run_config["name"],
        "scenario_spec": {
            "args": {
                "benchmark": benchmark,
                "subset": dataset.split(":")[-1] if ":" in dataset else None,
            }
        },
        "adapter_spec": {
            "model": run_config["model"]
        },
        "groups": [benchmark]
    }
    with open(f"{helm_output_path}/run_spec.json", "w", encoding="UTF-8") as f:
        json.dump(run_spec, f, indent=2, ensure_ascii=False)
    f.close()


def write_helm_output(stats: Dict[str, Any], helm_output_dir: str, task: str, model_name: str, benchmark:str):
    benchmark_dataSubSet_relation = {"benchmarkName": benchmark,
                   "dataSubSetList": []}
    for stat, dataset in stats:
        helm_output_path = f'{helm_output_dir}/{dataset}model={model_name.replace("/", "_")}'
        os.makedirs(helm_output_path, exist_ok=True)
        write_helm_stats(stat, helm_output_path, model_name, dataset)
        try:
            prompts = json.load(open(f"{helm_output_dir}/{task}/prompts.json", encoding="UTF-8"))
            generations = json.load(open(f"{helm_output_dir}/{task}/generations.json", encoding="UTF-8"))
            references = json.load(open(f"{helm_output_dir}/{task}/references.json", encoding="UTF-8"))
            cases = json.load(open(f"{helm_output_dir}/{task}/evaluation_cases.json", encoding="UTF-8"))
            write_helm_scenario_state(prompts, generations, references, helm_output_path, model_name)
            write_helm_per_instance_stats(prompts, generations, references, cases, helm_output_path, model_name)
        except Exception as e:
            pass
        benchmark_dataSubSet_relation["dataSubSetList"].append({
            "name": f'{dataset}model={model_name.replace("/", "_")}',
            "scenario_spec":{
                "args": {
                    "benchmark": dataset.split(":")[0],
                    "subset": dataset.split("-")[0].split(":")[-1] if ":" in dataset.split("-")[0] else None,
                }
            },
            "groups": [
                dataset.split("-")[0].split(":")[0]
            ]
        })
    with open(os.path.join(helm_output_dir, "benchmark_dataSubSet_relation.json"), "w") as file:
        json.dump(benchmark_dataSubSet_relation, file, indent=4)