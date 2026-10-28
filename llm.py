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