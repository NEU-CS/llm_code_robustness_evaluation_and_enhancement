import requests
import json
import time

class LocalClient():

    def __init__(self):
        pass

    def generate(self, ip, prompt, wait=False, **gen_kwargs):
        url = f"http://{ip}:8080"
        data = {
            "prompt": prompt,
            "max_new_tokens": gen_kwargs["max_length"],
            # "do_sample": gen_kwargs["do_sample"],
        }
        if gen_kwargs["do_sample"]:
            if "temperature" in gen_kwargs:
                data["temperature"] = gen_kwargs["temperature"]
            if "top_k" in gen_kwargs:
                data["top_k"] = gen_kwargs["top_k"]
            if "top_p" in gen_kwargs:
                data["top_p"] = gen_kwargs["top_p"]
        payload = json.dumps(data)
        headers = {
            'Content-Type': 'application/json'
        }
        cnt = 0
        total = 1
        if wait:
            total = 120
        while True:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                response = response.json()
                print(f"Server response: {response}")
                if "completions" in response and response["completions"] is not None:
                    return response["completions"]
                else:
                    return response["result"]
            except Exception as e:
                cnt = cnt + 1
                if cnt < total:
                    print(f"服务器ip: {ip}未返回结果或异常，若这是第一条请求，可能是服务还没启动成功等待1分钟后重试... 已重试次数:{cnt}/{total}", flush=True)
                    time.sleep(60)  # 等待60秒
                    continue
                raise e