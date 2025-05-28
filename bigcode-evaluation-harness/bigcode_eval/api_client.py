# encoding=utf-8
import json
import time
import openai
import logging
import requests

from typing import Any, Dict, List
from bigcode_eval.local_client import LocalClient
from bigcode_eval.ratelimit import sleep_and_retry, peak_rate_limit

class HttpAPIServer:
    def __init__(self, model_name: str, global_args: Dict):
        try:
            self.model_name = model_name.split("/")[1]
            self.global_args = global_args
            self.openai_api = ""
            self.deepseek_api = ""
            self.deepseek_url = "https://api.deepseek.com"

        except Exception as e:
            logging.exception(e)
            raise e

    # def init_glm(self):
    #     from .glm_client import GlmClient
    #     self.glm = GlmClient()
    #     # from .glm_6b_client import Glm6BClient
    #     # self.glm6b = Glm6BClient()

    # @staticmethod
    # def model_list() -> [str]:
    #     scales = [5, 20, 50, 200, 1000000]
    #     models = ["OpenAI/ChatGPT", "OpenAI/gpt4"]
    #     return [f"{model}[[EDS={scale}]]" for model in models for scale in scales]

    @staticmethod
    def name_match_list():
        return [
            "openai/chatgpt",
            "openai/gpt4",
            "openai/gpt-4",
            "openai/gpt-4o",
            "deepseek/deepseek-v2-chat",
        ]

    @staticmethod
    def use_api_client(model_name: str, global_args: Dict = None) -> bool:
        # if model_name in HttpAPIServer.model_list():
        #     return True
        for model in HttpAPIServer.name_match_list():
            if model.lower() in model_name.lower():
                return True
        if "api" in global_args and global_args["api"]:
            return True
        if "chat_mode" in global_args and global_args["chat_mode"] and global_args["chat_mode"] != "chat":
            return True
        if "customize_inference_ip" in global_args and global_args["customize_inference_ip"]:
            return True
        return False

    """
    提交请求 根据模型走不同的方式
    """
    def serve_request(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]
        # if "prompt_prefix" in global_args:
        #     prompt = global_args["prompt_prefix"] + prompt
        # if "prompt_suffix" in global_args:
        #     prompt = prompt + global_args["prompt_suffix"]
        raw_request["prompt"] = prompt
        # # 设置EOS配置
        # if "eos" in global_args:
        #     raw_request["stop_sequences"] = global_args['eos']

        if "api" in global_args and global_args["api"]:
            # if global_args["api"] == "cfs":
            #     return self.server_by_common_api(raw_request, global_args)
            if global_args["api"] == "ft":
                return self.server_by_ft_api(raw_request, global_args, **gen_kwargs)
        # elif "function_call" in global_args:
        #     if global_args["function_call"] == 'gpt3':
        #         return self.server_by_gpt_functional(raw_request, global_args, "gpt-3.5-turbo-16k-0613")
        #     else:
        #         return self.server_by_gpt_functional(raw_request, global_args, "gpt-4-0613")
        # elif "use_local_inference_file" in global_args:
        #     return self.server_by_local_inference_file(raw_request, global_args)
        elif "chat_mode" in global_args and global_args["chat_mode"]:
            ip = global_args["chat_mode"]
            return self.server_by_local_server(ip, raw_request, global_args, **gen_kwargs)
        elif "customize_inference_ip" in global_args and global_args["customize_inference_ip"]:
            ip = global_args["customize_inference_ip"]
            return self.server_by_local_server(ip, raw_request, global_args, wait=True, **gen_kwargs)
        elif "chatgpt" in self.model_name.lower():
            return self.server_by_chatgpt(raw_request, global_args, **gen_kwargs)
        elif "gpt4" in self.model_name.lower() or "gpt-4" in self.model_name.lower():
            return self.server_by_gpt4(raw_request, global_args, **gen_kwargs)
        elif "deepseek-v2" in self.model_name.lower():
            return self.server_by_deepseek_v2(raw_request, global_args)
        # elif "text-davinci-003" in self.model_name.lower() or \
        #         "text-ada-001" in self.model_name.lower() or \
        #         "gpt-3.5-turbo-instruct" in self.model_name.lower():
        #     return self.server_by_openai(raw_request)
        # elif "gpt-3.5-turbo-0613" in self.model_name.lower() or\
        #         "gpt-3.5-turbo-16k-0613" in self.model_name.lower():
        #     return self.server_by_openai_chat(raw_request)
        # elif "minimax" in self.model_name:
        #     return self.server_by_minimax(raw_request, global_args)
        # elif "chatglm2-66b" in self.model_name.lower():
        #     return self.server_by_chatglm2_66b(raw_request, global_args)
        # elif "baichuan2-53b" in self.model_name.lower() or "baichuan53b-api-v2" in self.model_name.lower():
        #     return self.server_by_baichuan2_53b(raw_request, global_args)
        # elif "chatglm-pro" in self.model_name.lower() or "glm-api-v2" in self.model_name.lower():
        #     return self.server_by_chatglm_pro(raw_request, global_args)
        # elif "spark-api-v2" in self.model_name.lower():
        #     return self.server_by_spark(raw_request, global_args)
        # elif "ernie" in self.model_name.lower():
        #     return self.server_by_ernie(raw_request, global_args)
        # else:
        #     return self.server_by_triton_glm(raw_request, global_args)

    """
    封装循环请求，防止限流和模型返回异常
    """
    def do_post(self, url, data, headers, model_name, prompt):
        # 发送POST请求
        response = requests.post(url, json=data, headers=headers)
        print(f"response.text: {response.text}")
        retry_times = 0
        while retry_times < 10:
            # print("网络返回失败：[" + response.status_code + "]" + response.message)
            # print("重试：" + str(data))
            text = json.loads(response.text)
            answer = text

            if response.status_code == 200 and answer is not None and len(answer) > 0:
                return answer

            print(f"触发限流or模型没有返回结果 重试次数:{retry_times} 等待10秒后重试...")
            time.sleep(20)  # 多等一会 一分钟最多请求100次

            retry_times += 1

            response = requests.post(url, data=data, headers=headers)
        raise Exception("OpenAI 没有返回结果")

    """
    获取sleep秒数 sleep到下一分钟+1秒
    """
    def get_sleep_interval(self):
        from datetime import datetime, timedelta
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        delta = next_minute - now
        seconds = delta.seconds
        return seconds + 1

    """
    封装gpt3.5模型循环请求
    """
    @sleep_and_retry
    @peak_rate_limit(calls=1000, peak_calls=5, period=60)
    # @Cat.transaction("Http", "do_post_chatgpt")
    def do_post_chatgpt(self, msg, model_name, **gen_kwargs):
        # 设置要提交的数据
        openai.api_key = self.openai_api  # 在speech平台申请的
        try:
            print(f"start query {model_name}..")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=msg,
                **gen_kwargs,
            )
            print(f"query {model_name} end..")
        except Exception as e:
            print(f"query {model_name} exception: {e}")
            response = None
        retry_times = 0
        while retry_times < 30:
            if response is not None and 'choices' in response and len(response['choices']) > 0:
                break
            retry_times = retry_times + 1
            time.sleep(self.get_sleep_interval()) # sleep到下一整分钟
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=msg,
                    **gen_kwargs,
                )
            except:
                response = None
        return response

    """
    封装gpt4模型循环请求
    """
    @sleep_and_retry
    @peak_rate_limit(calls=1000, peak_calls=5, period=60)
    # @Cat.transaction("Http", "do_post_gpt4")
    def do_post_gpt4(self, msg, model_name, **gen_kwargs):
        # 设置要提交的数据
        openai.api_key = self.openai_api
        try:
            print(f"start query {model_name}..")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=msg,
                **gen_kwargs,
            )
            print(f"query {model_name} end..")
        except Exception as e:
            print(f"query {model_name} exception: {e}")
            response = None
        retry_times = 0
        while retry_times < 10:
            if response is not None and 'choices' in response and len(response['choices']) > 0:
                break
            retry_times = retry_times + 1
            time.sleep(self.get_sleep_interval())  # rpm目前只有12，所以每次失败直接sleep1分钟防止浪费
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=msg,
                    **gen_kwargs,
                )
            except:
                response = None
        return response

    """
    请求gpt3的接口
    """
    def server_by_chatgpt(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]
        # 特殊处理 这个字符会让openai挂掉..替换一下
        prompt = prompt.replace("<|endoftext|>", "<|end_of_text|>")
        if len(prompt) == 0:
            print("prompt长度为0")
        
        # 设置要提交的数据
        msg = []
        if "system_cmd" in raw_request:
            msg.append({"role": "system", "content": raw_request["system_cmd"]})
        msg.append({"role": "user", "content": prompt})

        # model
        model_name = global_args["model_name"].lower().split("/")[-1].split("[[")[0]
        if "-" in model_name:
            model_name = model_name.lower().replace("chatgpt", "gpt-3.5-turbo")
        else:
            model_name = model_name.lower().replace("chatgpt", "gpt-3.5-turbo-0613")

        # 发送POST请求
        answer = self.do_post_chatgpt(msg, model_name, **gen_kwargs)

        # 处理结果
        try:
            if not isinstance(answer, str) and answer is not None:
                answer = answer["choices"][0]["message"]["content"]
            if answer is None:
                print(f"{model_name} 结果返回异常:{answer}, exception: None result")
                answer = f"{model_name} 结果返回异常"
        except Exception as e:
            print(f"{model_name} 结果返回异常:{answer}, exception:{e}")
            answer = f"{model_name} 结果返回异常"
        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": tokens,
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    请求gpt4的接口
    """
    def server_by_gpt4(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]

        msg = []
        if "system_cmd" in raw_request:
            msg.append({"role": "system", "content": raw_request["system_cmd"]})
        msg.append({"role": "user", "content": prompt})

        # model
        model_name = global_args["model_name"].lower().split("/")[-1].split("[[")[0]
        if "gpt-4-0125" in self.model_name.lower():
            model_name = "gpt-4-0125-preview"
        elif "gpt-4-turbo-2024-04-09" in self.model_name.lower() or "gpt-4-turbo-eva" in self.model_name.lower():
            model_name = "gpt-4-turbo-eva"
            gen_kwargs.pop("top_k", "top_k not found")
            gen_kwargs.pop("top_p", "top_p not found")
            gen_kwargs.pop("do_sample", "do_sample not found")
            gen_kwargs.pop("max_length", "max_length not found")
        elif "gpt-4o" in self.model_name.lower():
            model_name = "gpt-4o-2024-05-13"
            gen_kwargs.pop("top_k", "top_k not found")
            gen_kwargs.pop("top_p", "top_p not found")
            gen_kwargs.pop("do_sample", "do_sample not found")
            gen_kwargs.pop("max_length", "max_length not found")
        else:
            model_name = "gpt-4-0613"

        # 请求结果
        answer = self.do_post_gpt4(msg, model_name, **gen_kwargs)
        # 处理结果
        try:
            if not isinstance(answer, str) and answer is not None:
                answer = answer["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"{model_name} 结果返回异常:{answer}")
            answer = f"{model_name} 结果返回异常"

        if answer is None:
            answer = f"{model_name} 结果返回异常"

        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": [],
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    请求deepseek-v2的接口
    """
    def server_by_deepseek_v2(self, raw_request: Dict[str, Any], global_args):
        prompt = raw_request["prompt"]
        # prompt = "<|user|>\n" + prompt + "<|assistant|>\n"
        # 请求结果
        gen_kwargs = {
            "max_tokens": global_args["max_new_tokens"]
        }
        if global_args["temperature"] > 0:
            gen_kwargs["temperature"] = global_args["temperature"]
        if global_args["top_p"] > 0 and global_args["top_p"] < 1:
            gen_kwargs["top_p"] = global_args["top_p"]
        model_name = global_args["model_name"].lower().split("/")[-1].split("[[")[0]
        if "chat" in model_name:
            model_name = "deepseek-chat"
        else:
            model_name = "deepseek-coder"

        client = openai.OpenAI(api_key=self.deepseek_api,base_url=self.deepseek_url)
        msg = []
        msg.append({"role": "user", "content": prompt})
        print("debug information!!!")
        answer = client.chat.completions.create(
            model = model_name,
            messages = msg,
            **gen_kwargs,
        )

        answer = answer.choices[0].message.content
        # 处理结果
        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": [],
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}
    """
    自己起的服务调用
    """
    def server_by_local_server(self, ip, raw_request: Dict[str, Any], global_args, wait=False, **gen_kwargs):
        prompt = raw_request["prompt"]
        # 请求结果
        try:
            answer = LocalClient().generate(ip, prompt, wait=wait, **gen_kwargs)
        except Exception as e:
            print(f"HTTP API 无结果返回: {e}")
            answer = "<<HTTP API 无结果返回>>"
            raise e
        # 处理结果
        completions = []
        tokens = []
        if isinstance(answer, list):
            completions = answer
        else:
            completions.append(
                {
                    "text": answer,
                    "tokens": [],
                    "logprobs": [0.0] * len(tokens),
                    "top_logprobs_dicts": [{token: 0.0} for token in tokens]
                }
            )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}