import base64
import io
import json
import os
import re

import requests
from PIL import Image, ImageDraw

API_URL = "https://api.meta.ai/v1/responses"
BOX_RE = re.compile(r"<\|box;([^|]*)\|>")


def _to_data_url(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sam_find(rgb, phrase):
    """Ask SAM 3.1 for every match of a short noun phrase. Returns a list of boxes."""
    body = {
        "model": "sam-3.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": _to_data_url(rgb)},
            ],
        }],
        "stream": True,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
    }
    r = requests.post(
        API_URL,
        headers={
            "Authorization": "Bearer " + os.environ["MODEL_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json=body, stream=True, timeout=90,
    )
    if r.status_code != 200:
        raise RuntimeError(f"SAM error {r.status_code}: {r.text[:300]}")

    text = ""
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("type") == "response.output_text.delta":
            text += event.get("delta", "")

    boxes = []
    for m in BOX_RE.finditer(text):
        f = dict(kv.split("=") for kv in m.group(1).split(";"))
        x1, y1, x2, y2 = (int(f[k]) for k in ("x1", "y1", "x2", "y2"))
        boxes.append({
            "box": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "conf": float(f["c"]) if "c" in f else None,
        })
    return boxes


def draw_boxes(rgb, results, path):
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for label, boxes in results.items():
        for b in boxes:
            d.rectangle(b["box"], outline=(0, 255, 0), width=2)
            d.text((b["box"][0], b["box"][1] - 12), label, fill=(0, 255, 0))
    img.save(path)