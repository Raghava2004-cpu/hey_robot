import base64
import io
import os

import numpy as np
import requests
from openai import OpenAI
from PIL import Image

from robot_env import W, H

_client = None

PROMPT = ("This is a top-down robot camera view of a table. Show the scene AFTER this task "
          "is done: '{task}'. Change only what the task requires. Keep everything else exactly "
          "the same: same camera angle, same other objects and positions, same robot arm, "
          "same lighting.")


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    return _client


def imagine(task, rgb):
    """Muse Image edits the current camera view into the predicted goal scene."""
    import home
    task = home.to_display(task)           # --home: "station C" -> "the bedside table", "warehouse" -> "home"
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    buf.name = "scene.png"
    resp = _get_client().images.edit(model="muse-image-1.0",
                                     prompt=PROMPT.format(task=task), image=buf, n=1)
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        img = Image.open(io.BytesIO(base64.b64decode(item.b64_json)))
    else:
        img = Image.open(io.BytesIO(requests.get(item.url, timeout=60).content))
    return np.array(img.convert("RGB").resize((W, H)))