import os
import re

import requests
import time


def post_retry(*args, **kwargs):
    for i in range(4):
        try:
            return requests.post(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if i == 3:
                raise
            time.sleep(5 * (i + 1))  # wait 5s, 10s, 15s, then try again

CHAT_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3-contributor"


def spark(messages, effort="low", model=MODEL):
    body = {"model": model, "messages": messages}
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"],
               "Content-Type": "application/json"}
    r = post_retry(CHAT_URL, headers=headers, json=body, timeout=240)
    if r.status_code == 400 and effort:  # model may not accept reasoning_effort
        body.pop("reasoning_effort")
        r = post_retry(CHAT_URL, headers=headers, json=body, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"{model} error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()
def spark_vision(question, rgb):
    from perception import _to_data_url
    messages = [
        {"role": "system", "content":
            "You see a robot's top-down camera image of a table. Answer in a few words only, "
            "using short noun phrases like 'yellow cube'. If unsure, answer 'unknown'."},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": _to_data_url(rgb)}},
        ]},
    ]
    answer, _ = spark(messages, effort="low")
    return answer.strip().strip(".").lower()