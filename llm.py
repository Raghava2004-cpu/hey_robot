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

def spark_compare(task, imagined, real):
    """Compare the imagined goal with the real result. Returns (match, reason)."""
    from perception import _to_data_url
    messages = [
        {"role": "system", "content":
            "You verify a robot's work. Image 1 is the IMAGINED goal scene. Image 2 is the REAL "
            "camera view after the robot acted. Ignore the robot arm's pose, small position "
            "offsets, image style and resolution, and objects the task does not mention (they may "
            "end up in different places). Judge only whether the real scene achieves "
            "the task the way the goal shows. An object resting on top of a robot or platform counts "
            "as landed or placed on it, even if it looks raised. The imagined goal is a drawing: its objects "
            "may have a different color, size or style than the real ones, so count an object as present when "
            "an object of the same kind is on the same table or in the same container (its exact spot there does not "
            "matter). Judge only what this camera can see: a "
            "place from the task that is outside the picture is not missing. Reply with MATCH or MISMATCH on "
            "the first line, then one short reason."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Task: {task}"},
            {"type": "image_url", "image_url": {"url": _to_data_url(imagined)}},
            {"type": "image_url", "image_url": {"url": _to_data_url(real)}},
        ]},
    ]
    answer, _ = spark(messages, effort="low")
    lines = answer.strip().splitlines()
    first = lines[0].upper() if lines else ""
    match = "MATCH" in first and "MISMATCH" not in first
    reason = " ".join(lines[1:]).strip() or first
    return match, reason