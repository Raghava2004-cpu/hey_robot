"""Show and tell: hold up a REAL object to the webcam and say "Bring me one of these."

  1. The webcam picture is taken while you speak.
  2. Muse Spark names what you are holding and picks the matching warehouse product
     (a real apple -> "we have no apples; the closest product is the pear").
  3. SAM 3.1 draws the box around it (saved in runs/real/show_XX.png and added to the Muse Brain feed).
  4. "these / this / it" in your order becomes the product ("Bring me the soup can"), and the order goes
     through the normal pipeline: SafetyShield -> Spark router -> robots -> checks.

Used by:  python command_center.py --loop --home --webcam
"""
import os
import re

import numpy as np

PRODUCTS = ["banana", "pear", "strawberry", "scissors", "medium clamp", "soup can", "gelatin box",
            "foam brick", "mustard bottle"]
import home as _home
PRODUCTS += _home.items()         # home mode: water bottle, notebook, pen
POINTING = re.compile(r"\b((this|that|these|those) (kind|type|sort) of \w+|things like (this|these)|"
                      r"(this|that|these|those) (items?|objects?|things?|products?)|"
                      r"one of these|one of those|some of these|more of these|these ones|those ones|this one|"
                      r"that one|this thing|that thing|like this|like these|these|those|this|that|it)\b", re.I)
_cam = {"cap": None, "n": 0}


def open_camera():
    import cv2
    if _cam["cap"] is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("no webcam found")
        _cam["cap"] = cap
    return _cam["cap"]


def preview(seconds=2.0, text="Hold the object up to the webcam"):
    """Show the live webcam for a moment so you can aim the object. The last clean frame is kept: it is taken
    while you hold the object up, before you start speaking."""
    import time
    import cv2
    cap = open_camera()
    t = time.time()
    while time.time() - t < seconds:
        ok, frame = cap.read()
        if ok:
            _cam["preview"] = frame[:, :, ::-1].copy()          # RGB, without the text drawn on it
            shown = frame.copy()
            cv2.putText(shown, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Hey Robot show and tell (webcam)", shown)
            cv2.waitKey(30)


def last_preview():
    return _cam.get("preview")


def grab():
    """A fresh webcam picture as RGB (the camera buffers old frames, so read a few)."""
    cap = open_camera()
    frame = None
    for _ in range(6):
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        raise RuntimeError("could not read the webcam")
    return frame[:, :, ::-1].copy()


def identify(rgb):
    """Muse Spark: what is the person holding up, and which warehouse product matches it?
    (Its own prompt: the general vision helper is told it sees a top-down table and may answer 'unknown'.)"""
    from llm import spark
    from perception import _to_data_url
    q = ("This webcam picture shows a person holding up ONE object to show a robot what they want. "
         "Name that object (for example: scissors, banana, soup can, apple, phone, mug). Then choose the most "
         f"similar product from this list: {', '.join(PRODUCTS)}, or none. Never answer 'unknown': if you are not "
         "sure, give your best guess. Reply in exactly this format: "
         "OBJECT: <short name> | PRODUCT: <product from the list or none> | WHY: <a few words>")
    messages = [{"role": "system", "content": "You are the eyes of a warehouse robot. Look carefully at the picture."},
                {"role": "user", "content": [{"type": "text", "text": q},
                                             {"type": "image_url", "image_url": {"url": _to_data_url(rgb)}}]}]
    text, _ = spark(messages, effort="low")
    text = str(text)
    obj = re.search(r"object:\s*([^|]+)", text, re.I)
    prod = re.search(r"product:\s*([^|]+)", text, re.I)
    why = re.search(r"why:\s*(.+)", text, re.I)
    obj = obj.group(1).strip().lower().strip(" .") if obj else text.strip().lower()[:30]
    prod = prod.group(1).strip().lower().strip(" .") if prod else "none"
    product = next((p for p in PRODUCTS if p in prod), None) or next((p for p in PRODUCTS if p in obj), None)
    return obj, product, (why.group(1).strip() if why else "")


def resolve(order, frames):
    """frames: webcam pictures (taken before and after you spoke). The first one Spark can name is used.
    Returns (new order or None, spoken line, info). New order None = nothing to fetch."""
    import muse_eye
    from perception import sam_find
    frames = [f for f in (frames if isinstance(frames, (list, tuple)) else [frames]) if f is not None]
    obj, product, why, rgb = "nothing", None, "", frames[-1]
    for f in frames:
        obj, product, why = identify(f)
        rgb = f
        if product:
            break
    boxes = []
    for ph in [obj] + ([product] if product and product != obj else []):
        boxes = sam_find(rgb, ph) if ph else []
        if boxes:
            break
    boxes = sorted(boxes, key=lambda b: -(b.get("conf") or 0))[:1]
    os.makedirs("runs/real", exist_ok=True)
    _cam["n"] += 1
    path = f"runs/real/show_{_cam['n']:02d}.png"
    title = f"SHOW AND TELL | Spark: {obj} -> product: {product or 'none'} | SAM 3.1 box"
    muse_eye.annotate(rgb, [dict(b, label=obj) for b in boxes], title, chosen=0).save(path)
    muse_eye.event("SAM 3.1", f"webcam: {obj} {(boxes[0].get('conf') or 0):.2f}" if boxes else f"webcam: {obj} (no box)")
    muse_eye.event("SPARK", f"you showed: {obj} -> product: {product or 'none'}")
    info = {"object": obj, "product": product, "why": why, "box": boxes[0]["box"] if boxes else None, "picture": path}
    if not product:
        return None, f"I can see {'an' if obj[:1] in 'aeiou' else 'a'} {obj}, but we don't stock anything like it.", info
    same = product in obj or obj in product
    thing = "a pair of scissors" if obj.endswith("scissors") else f"{'an' if obj[:1] in 'aeiou' else 'a'} {obj}"
    line = (f"You showed me {thing}." if same else
            f"You showed me {thing}. We don't stock that, so I'll use the closest product, the {product}.")
    if POINTING.search(order):
        new = POINTING.sub(f"the {product}", order, count=1)
    else:
        new = f"{order} (the object I am showing: the {product})"
    return new, line, info
