"""Runs every skill once with a scripted plan (no Spark planning) and prints PASS / FAIL.
Uses the real SAM 3.1 and Muse Spark checks, so keep the hotspot on.

python test_all_skills.py               # all 9 jobs with the 3D window
python test_all_skills.py throw land    # only some skills
python test_all_skills.py --fast        # no window (faster)
"""
import os
import sys
import time

import pybullet as p

from skill_tasks import run_plan, task_for
from team_skills import MAP_FILE

ORDERS = [("deliver", "Bring the banana and the soup can to the packing station.", ["banana", "soup can"], None),
          ("move", "Move the banana to station C.", ["banana"], "station C"),
          ("throw", "Throw a tennis ball into the blue bin.", None, None),
          ("land", "Land the drone on the moving robot.", None, None),
          ("audit", "Check the inventory and fix what is wrong.", None, None),
          ("deliver", "I'm hungry, bring me some fruit.", ["banana", "pear", "strawberry"], None),
          ("sort", "Fruit in the tray, tools on station A.", None, None),
          ("air", "Drone, get the strawberry and the pear from the top shelf.", ["strawberry", "pear"], None),
          ("relay", "Get the strawberry from the top shelf and put it on station C.", ["strawberry"], "station C")]


def run_all(only=(), gui=True, on_build=None):
    results = []
    for skill, order, items, place in ORDERS:
        if only and skill not in only:
            continue
        if os.path.exists(MAP_FILE):
            os.remove(MAP_FILE)
        print(f"\n===== {skill.upper()}: \"{order}\" =====")
        t = task_for(skill, order, items, place)
        env = t["scene"](gui=gui)
        if on_build:
            on_build(env)
        api = t["api_cls"](env)
        t0 = time.time()
        err = ""
        try:
            run_plan(skill, api, items, place)
        except Exception as e:
            err = f"  CRASHED: {type(e).__name__}: {e}"
        env.step(240)
        ok, msg = t["check"](env)
        for s in api.log:
            print("   ", s)
        print(f"CHECK: {ok} - {msg}{err}")
        results.append((skill, ok, msg + err, round(time.time() - t0)))
        p.disconnect()
    print("\n=========== SUMMARY ===========")
    for skill, ok, msg, secs in results:
        print(f"{'PASS' if ok else 'FAIL'}  {skill:8s} {secs:4d} s  {msg}")
    return results


if __name__ == "__main__":
    run_all([a for a in sys.argv[1:] if not a.startswith("--")], gui="--fast" not in sys.argv)
