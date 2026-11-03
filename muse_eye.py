"""Muse in the loop, like a Roboflow / YOLO app: the robots ask the vision models before every action.

  1. SAM 3.1 finds the thing in the camera picture (box + confidence).
  2. The depth image turns the box into 3D coordinates (where the hand or the drone must go).
  3. Muse Spark looks at the crop and confirms WHAT it is before the arm moves.
  4. Every detection is saved as a labelled picture + one JSON line in runs/vision/ (the evidence).

The control room (GUI runs only) is a live 2x2 window: robot front, wrist, drone and ceiling cameras.
When SAM runs, that camera's tile freezes for 2 s on the labelled detection, like a vision app.
"""
import json
import math
import os
import time

import numpy as np

DIR = "runs/vision"
LAST = {}                 # camera -> {"img": labelled picture, "t": sim time, "text": summary}
STATE = {"step": 0, "action": "", "sim_t": 0.0, "save_png": True,
         "order": "", "code": [], "line": -1, "events": [], "counts": {}}
TAG_COLORS = {"VOICE": (120, 200, 255), "SAFETY": (255, 120, 120), "SPARK": (255, 200, 80),
              "WONDER": (220, 140, 255), "IMAGE": (255, 150, 210), "SAM 3.1": (110, 230, 140),
              "ROBOT": (200, 200, 200), "DRONE": (140, 220, 255), "CHECK": (255, 255, 255)}


def event(tag, text):
    """One line in the Muse Brain live feed."""
    STATE["events"].append((tag, str(text)[:70]))
    del STATE["events"][:-12]
    STATE["counts"][tag] = STATE["counts"].get(tag, 0) + 1


def set_order(text):
    STATE["order"] = str(text)
    STATE["events"].clear()
    STATE["counts"].clear()
    event("VOICE", f'heard: "{str(text)[:58]}"')


def set_code(code):
    STATE["code"] = str(code).splitlines()
    STATE["line"] = -1
    event("SPARK", f"wrote the robot program ({len(STATE['code'])} lines)")


def on_action(action, arg, result):
    """Called after every robot action: highlight the program line that called it, add it to the feed."""
    code, start = STATE["code"], STATE["line"] + 1
    for k in list(range(start, len(code))) + list(range(0, start)):
        if f"robot.{action}(" in code[k]:
            STATE["line"] = k
            break
    tag = "DRONE" if action.startswith("drone") or action == "handoff_to_robot" else "ROBOT"
    ok = not str(result).lower().startswith(("could not", "refused", "failed", "nothing", "not "))
    event(tag, f"{action}({arg}) {'ok' if ok else 'FAILED'}: {str(result)[:40]}")
    set_action(f"{action}({arg}) -> {str(result)[:50]}")
CATALOG = ["banana", "pear", "strawberry", "scissors", "medium clamp", "soup can", "gelatin box",
           "foam brick", "mustard bottle", "tennis ball"]
import home as _home
CATALOG += _home.items()          # home mode: water bottle, notebook, pen


def set_action(text):
    STATE["action"] = str(text)


def _font(size=14):
    from PIL import ImageFont
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _font_mono(size=12):
    from PIL import ImageFont
    for f in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _model_counts():
    """Calls per Meta model so far in this job (from usage_meter when it is installed)."""
    try:
        import usage_meter
        if usage_meter.STATS:
            short = {"SAM 3.1": "SAM", "Muse Spark": "Spark", "Muse Wonder": "Wonder", "Muse Image": "Image",
                     "Voice Transcribe": "Voice"}
            return {short.get(n, n): s["calls"] for n, s in usage_meter.STATS.items()}
    except Exception:
        pass
    c = STATE["counts"]
    return {k: v for k, v in c.items() if k in ("SAM 3.1", "SPARK", "IMAGE", "WONDER")}


def annotate(rgb, dets, title="", chosen=0):
    """Boxes + labels on the picture (returns a PIL image)."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    f = _font(15)
    for i, det in enumerate(dets):
        x0, y0, x1, y1 = [float(v) for v in det["box"]]
        col = (0, 230, 90) if i == chosen else (255, 190, 0)
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f'{det.get("label", "")} {det.get("conf") or 0:.2f}'
        if det.get("xyz") is not None:
            label += " ({:.2f}, {:.2f}, {:.2f})".format(*det["xyz"])
        tw = d.textlength(label, font=f) if hasattr(d, "textlength") else 8 * len(label)
        ty = max(0, y0 - 19)
        d.rectangle([x0, ty, x0 + tw + 6, ty + 18], fill=col)
        d.text((x0 + 3, ty + 1), label, fill=(0, 0, 0), font=f)
    if title:
        d.rectangle([0, 0, img.width, 22], fill=(15, 20, 30))
        d.text((6, 3), title, fill=(255, 255, 255), font=f)
    return img


def log(camera, prompt, rgb, dets, chosen=0, xyz=None, spark=None, model="SAM 3.1", save=True):
    """Record one detection: labelled picture + JSON line + live tile for the control room."""
    STATE["step"] += 1
    n = STATE["step"]
    dets = [dict(dt, label=dt.get("label", prompt)) for dt in dets]
    if dets and xyz is not None:
        dets[chosen]["xyz"] = [round(float(v), 3) for v in xyz]
    head = f"#{n} {camera.upper()} | {model}: '{prompt}' -> {len(dets)} found"
    if spark is not None:
        head += f" | Muse Spark: {spark}"
    img = annotate(rgb, dets, head, chosen)
    LAST[camera] = {"img": img, "t": STATE["sim_t"], "text": head}
    if dets:
        cf = dets[chosen].get("conf")
        best = f"{dets[chosen].get('label', prompt)}" + (f" {cf:.2f}" if cf else "")
    else:
        best = "nothing"
    event("SAM 3.1", f"{camera} cam '{prompt}': {best}" + (f" | Spark: {spark}" if spark else ""))
    rec = {"step": n, "time": round(time.time(), 2), "action": STATE["action"], "camera": camera,
           "model": model, "prompt": prompt, "found": len(dets),
           "boxes": [[round(float(v)) for v in dt["box"]] for dt in dets],
           "conf": [round(float(dt.get("conf") or 0), 3) for dt in dets],
           "chosen": chosen if dets else None, "xyz": dets[chosen].get("xyz") if dets else None,
           "spark": spark}
    try:
        os.makedirs(DIR, exist_ok=True)
        if save and STATE["save_png"]:
            img.save(os.path.join(DIR, f"{n:04d}_{camera}_{prompt.replace(' ', '_')}.png"))
            rec["image"] = f"{n:04d}_{camera}_{prompt.replace(' ', '_')}.png"
        with open(os.path.join(DIR, "detections.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def confirm(rgb, box, name, catalog=None):
    """Muse Spark names the product in the box (crop with a margin). Returns (is_it, answer)."""
    try:
        from llm import spark_vision
    except Exception:
        return True, "unavailable"
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = np.asarray(rgb).shape[:2]
    m = max(12, int(0.25 * max(x1 - x0, y1 - y0)))
    crop = np.asarray(rgb)[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    names = sorted(set((catalog or CATALOG) + [name]))
    q = (f"This is a close crop from a warehouse robot camera. Which product is in the middle of it? "
         f"Choose exactly one: {', '.join(names)}, other. Reply with the product name only.")
    try:
        ans = str(spark_vision(q, crop)).strip().lower().strip("*\"'. ")
    except Exception:
        return True, "unavailable"
    return (name in ans or ans.startswith("yes")), ans[:30]


def detect_and_confirm(rgb, phrases, name, to_xyz, camera, verify=True, max_check=3):
    """SAM 3.1 boxes for the first phrase that finds something; Spark confirms the best box (tries up to
    3 boxes). to_xyz(det) -> 3D point. Returns (det or None, xyz, spark answer)."""
    from perception import sam_find
    boxes, used = [], phrases[0]
    for ph in phrases:
        boxes = sam_find(rgb, ph)
        if boxes:
            used = ph
            break
    boxes = sorted(boxes, key=lambda b: -(b.get("conf") or 0))
    if not boxes:
        log(camera, used, rgb, [], spark=None)
        return None, None, None
    chosen, answer = 0, None
    if verify:
        for i, b in enumerate(boxes[:max_check]):
            ok, answer = confirm(rgb, b["box"], name)
            if ok:
                chosen = i
                break
        else:
            log(camera, used, rgb, boxes, chosen=0, spark=f"{answer} (not a {name})")
            return None, None, answer
    xyz = to_xyz(boxes[chosen])
    log(camera, used, rgb, boxes, chosen=chosen, xyz=xyz, spark=answer)
    return boxes[chosen], xyz, answer


# ---------- the control room: 4 live cameras in one window (GUI runs only) ----------
class ControlRoom:
    TILE = (400, 300)

    def __init__(self, env, every=60, save_frames=True):
        self.env, self.every, self.k = env, every, 0
        self.save_frames = save_frames
        self.frame = 0
        self.cv2 = None
        headless = os.name != "nt" and not os.environ.get("DISPLAY")
        if os.environ.get("SKILLFORGE_NO_WINDOW") != "1" and not headless:
            try:
                import cv2                                  # pip install opencv-python (for the live window)
                self.cv2 = cv2
            except Exception:
                print("control room: no live window (pip install opencv-python); frames are still saved")
        os.makedirs("runs/control_room", exist_ok=True)
        env.hooks.append(self.tick)

    def _render(self, eye, target, up, fov=60):
        import pybullet as p
        w, h = self.TILE
        view = p.computeViewMatrix(list(eye), list(target), list(up))
        proj = p.computeProjectionMatrixFOV(fov, w / h, 0.03, 8.0)
        out = p.getCameraImage(w, h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               flags=p.ER_NO_SEGMENTATION_MASK)
        return np.reshape(out[2], (h, w, 4))[:, :, :3].astype(np.uint8)

    def _views(self):
        import pybullet as p
        env = self.env
        x, y, yaw = env.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        views = {"front": ([x + 0.5 * c, y + 0.5 * s, 1.0], [x + 0.85 * c, y + 0.85 * s, 0.45], [0, 0, 1], 60)}
        from mobile_env import EE_LINK
        hp, ho = p.getLinkState(env.panda, EE_LINK)[4:6]
        R = np.array(p.getMatrixFromQuaternion(ho)).reshape(3, 3)
        eye = np.array(hp) + R @ np.array([0.0, 0.0, -0.02])
        views["wrist"] = (eye, eye + R @ np.array([0, 0, 0.4]), R @ np.array([1.0, 0, 0]), 75)
        d = getattr(env, "drone", None)
        if d is not None:
            dx, dy, dz = p.getBasePositionAndOrientation(d.body)[0]
            views["drone"] = ([dx, dy, dz - 0.2], [dx, dy, 0.0], [1, 0, 0], 70)
        views["ceiling"] = ([3.4, -3.4, 3.6], [0.0, 0.0, 0.2], [0, 0, 1], 55)
        return views

    PANEL = 460

    def _brain(self, d, x0, height):
        """The MUSE BRAIN panel: the order, model calls, Spark's program (current line lit), live feed."""
        mono = _font_mono(12)
        d.rectangle([x0, 0, x0 + self.PANEL, height], fill=(16, 20, 30))
        d.text((x0 + 10, 8), "MUSE BRAIN  (Meta models at work)", fill=(255, 210, 90), font=_font(16))
        y = 34
        import home
        order = home.to_display(STATE["order"]) or "(waiting for a voice order)"
        for k in range(0, min(len(order), 116), 58):
            d.text((x0 + 10, y), ("ORDER: " if k == 0 else "       ") + order[k:k + 58], fill=(200, 230, 255), font=mono)
            y += 15
        if home.active():                    # the program keeps the inside names: show how they map
            for line in home.LEGEND:
                d.text((x0 + 10, y), line[:62], fill=(255, 170, 120), font=mono)
                y += 15
        counts = _model_counts()
        d.text((x0 + 10, y + 4), "  ".join(f"{n} {c}" for n, c in counts.items()) or "no model calls yet",
               fill=(150, 255, 170), font=mono)
        y += 26
        d.text((x0 + 10, y), "PROGRAM written by Muse Spark (running line lit):", fill=(255, 200, 80), font=mono)
        y += 16
        code, cur = STATE["code"], STATE["line"]
        first = max(0, min(cur - 6, len(code) - 16)) if code else 0
        for k in range(first, min(len(code), first + 16)):
            if k == cur:
                d.rectangle([x0 + 6, y - 1, x0 + self.PANEL - 6, y + 13], fill=(30, 110, 60))
            d.text((x0 + 10, y), f"{k + 1:3d} {code[k][:56]}", fill=(255, 255, 255) if k == cur else (170, 180, 195),
                   font=mono)
            y += 14
        y = max(y + 8, height - 12 * 15 - 24)
        d.text((x0 + 10, y), "LIVE FEED", fill=(255, 200, 80), font=mono)
        y += 16
        for tag, text in STATE["events"][-12:]:
            d.text((x0 + 10, y), f"[{tag}]", fill=TAG_COLORS.get(tag, (220, 220, 220)), font=mono)
            d.text((x0 + 92, y), home.to_display(text)[:52], fill=(220, 225, 235), font=mono)
            y += 15

    def tick(self):
        self.k += 1
        STATE["sim_t"] = self.k / 240
        if self.k % self.every:
            return
        from PIL import Image, ImageDraw
        names = {"front": "ROBOT FRONT CAM", "wrist": "WRIST CAM", "drone": "DRONE CAM", "ceiling": "CEILING CAM"}
        tiles = []
        for key, (eye, target, up, fov) in self._views().items():
            last = LAST.get(key) or (LAST.get("basket") if key == "wrist" else None)
            if last and STATE["sim_t"] - last["t"] < 2.0:          # freeze on the latest detection
                img = last["img"].resize(self.TILE)
            else:
                img = Image.fromarray(self._render(eye, target, up, fov))
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, img.height - 20, img.width, img.height], fill=(15, 20, 30))
            dr.text((6, img.height - 18), names.get(key, key), fill=(120, 220, 255), font=_font(13))
            tiles.append(img)
        while len(tiles) < 4:
            tiles.append(Image.new("RGB", self.TILE, (20, 20, 20)))
        w, h = self.TILE
        board = Image.new("RGB", (2 * w + self.PANEL, 2 * h + 34), (10, 12, 18))
        for i, t in enumerate(tiles[:4]):
            board.paste(t, ((i % 2) * w, 34 + (i // 2) * h))
        d = ImageDraw.Draw(board)
        import home
        d.text((8, 8), f"HEY ROBOT CONTROL ROOM  |  {home.to_display(STATE['action'])[:60]}", fill=(255, 255, 255), font=_font(16))
        self._brain(d, 2 * w, board.height)
        arr = np.asarray(board)
        if self.save_frames:
            self.frame += 1
            try:
                board.save(f"runs/control_room/{self.frame:05d}.jpg", quality=80)
            except Exception:
                pass
        if self.cv2 is not None:
            try:
                self.cv2.imshow("Hey Robot control room", arr[:, :, ::-1])
                self.cv2.waitKey(1)
            except Exception:
                self.cv2 = None
