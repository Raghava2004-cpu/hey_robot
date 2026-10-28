import os
import re

import requests

CHAT_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3-contributor"


def spark(messages, effort="low"):
    body = {"model": MODEL, "messages": messages}
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"],
               "Content-Type": "application/json"}
    r = requests.post(CHAT_URL, headers=headers, json=body, timeout=120)
    if r.status_code == 400 and effort:  # model may not accept reasoning_effort
        body.pop("reasoning_effort")
        r = requests.post(CHAT_URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Spark error {r.status_code}: {r.text[:300]}")
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