"""Voice questions: "How many items are on station C?", "What is on station B?", "Where is the pear?"
The drone flies there and looks; Muse Spark answers from the drone's picture (SAM 3.1 finds items for
"where is" questions). The simulator's ground truth is printed next to the answer, so it can be checked."""
import math
import os
import re

import pybullet as p

from audit_tasks import at_station
from store_tasks import store_scene
from team_skills import TeamAPI

WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
PLACES = ["station A", "station B", "station C"]


def truth_on(env, place):
    return [n for n in env.objects if n not in env.static and n != "drone" and at_station(env, n, place)]


def first_number(text):
    for w in re.findall(r"[a-z]+|\d+", text.lower()):
        if w.isdigit():
            return int(w)
        if w in WORDS:
            return WORDS[w]
    return None


def look_at(env, place):
    """Drone hovers 0.8 m in front of the table, 1.1 m high, and takes a picture of it."""
    d = env.drone
    cx, cy = env.station_centers[place]
    yaw = env.station_yaws[place]
    d.fly_to(cx - 0.8 * math.cos(yaw), cy - 0.8 * math.sin(yaw), 1.1, 3.0)
    rgb, _ = d.camera_look([cx, cy, 0.47])
    os.makedirs("runs", exist_ok=True)
    path = f"runs/ask_{place.replace(' ', '_')}.png"
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception:
        path = None
    return rgb, path


def answer(env, question, place=None, items=()):
    """Returns (answer text, ground truth text, correct or None)."""
    from llm import spark_vision
    from mission_tasks import YCB_FILES
    if place in PLACES:
        rgb, path = look_at(env, place)
        truth = truth_on(env, place)
        q = (f"You are the eyes of a warehouse drone. The picture shows the table of {place}, seen from the "
             f"drone camera in a simple 3D simulation (objects look plain; judge by shape and color). "
             f"Products the warehouse stocks: {', '.join(sorted(YCB_FILES))}. "
             f"Question from the operator: \"{question}\" Answer in one short sentence. "
             "If the question asks how many, start with the number.")
        ans = spark_vision(q, rgb).strip()
        ok = None
        if "how many" in question.lower():
            n = first_number(ans)
            ok = n == len(truth)
        return ans, f"really on {place}: {truth} ({len(truth)} items); picture: {path}", ok
    # "where is X?": the drone scans every station with SAM 3.1 + Spark's product check
    api = TeamAPI(env)
    found = {}
    for st in PLACES:
        for it in api.drone_scan(st, list(items)):
            found[it] = st
    ans = "; ".join(f"the {it} is on {found[it]}" if it in found else f"I could not find the {it}"
                    for it in items) or "Tell me which item to look for."
    truth = {it: next((st for st in PLACES if at_station(env, it, st)), "nowhere") for it in items}
    ok = all(found.get(it, "nowhere") == truth[it] for it in items) if items else None
    return ans, f"really: {truth}", ok


def run_ask(question, place=None, items=(), gui=True):
    env = store_scene()(gui=gui)
    ans, truth, ok = answer(env, question, place, items)
    print(f"\n🛸 ANSWER: {ans}")
    try:
        from speech import say
        say(ans, wait=True)
    except Exception:
        pass
    print(f"   ({truth})")
    if ok is not None:
        print(f"   correct: {ok}")
    env.say(ans[:60])
    env.step(480)
    p.disconnect()
    return bool(ok) if ok is not None else True
