"""Sim-to-sim test: the RL reflex (trained on a simple GPU simulator) flies the PyBullet drone.
Compares 3 landings on the moving robot under hidden wind + random gusts:
  open-loop (aim once)  |  camera tracking (SAM every 0.25 s)  |  RL reflex (PPO policy)
Default perception = the simulator's object mask (fast). Add --sam to use real SAM 3.1 (slow, costs API calls).

python exp_rl_landing.py              # 20 conditions x 3 methods, mask detector (about 5-10 min)
python exp_rl_landing.py --sam --n 4  # spot check with real SAM 3.1
"""
import json
import math
import sys

import numpy as np
import pybullet as p

from drone import Drone
from mobile_env import MobileEnv

ARGS = sys.argv[1:]
USE_SAM = "--sam" in ARGS
N = int(ARGS[ARGS.index("--n") + 1]) if "--n" in ARGS else 20
GUSTS = [0.0, 0.5, 1.0, 1.5]          # gust strength in m/s^2 (1.5 = stronger than most training)
METHODS = ["open_loop", "tracking", "rl"]


def condition(i):
    rng = np.random.default_rng(1000 + i)
    wm, wa = rng.uniform(0, 1.2), rng.uniform(0, 2 * math.pi)
    return {"seed": i, "wind": [round(wm * math.cos(wa), 3), round(wm * math.sin(wa), 3)],
            "gust": GUSTS[i % len(GUSTS)], "wheel": round(float(rng.uniform(0.6, 2.2)), 2)}


def run(c, method):
    env = MobileEnv(gui=False)
    env.add_landing_pad()
    d = Drone(env, start=(-0.6, 0.6, 1.3), wind=c["wind"])
    d.detector = "sam" if USE_SAM else "mask"
    d.set_gusts(c["gust"], seed=c["seed"])
    env.step(120)
    # the planner's part, exactly as in the missions: one hover to measure the wind, two pad sightings
    env.set_wheels(c["wheel"], c["wheel"])
    gx, gy, _ = env.pose()
    pos = d.fly_to(gx, gy, 1.3)
    d.set_feedforward(-6 * (pos[0] - gx), -6 * (pos[1] - gy))
    p1 = d.find("landing pad")
    gx, gy, _ = env.pose()
    d.fly_to(gx - 0.38 + 0.6, gy, 1.3)
    p2 = d.find("landing pad")
    if p1 and p2:
        vx, vy = (p2[0] - p1[0]) / 3.0, (p2[1] - p1[1]) / 3.0
    else:
        vx, vy = 0.2, 0.0
    if method == "open_loop":
        aim = (p2 or p1 or [gx - 0.38, gy])
        ok = d.land_at(aim[0] + vx * 2.0, aim[1] + vy * 2.0)
    elif method == "tracking":
        ok = d.land_tracking(vx, vy)
    else:
        ok = d.land_rl(vx, vy)
    miss = d.touchdown_miss
    p.disconnect()
    return {"landed": bool(ok), "cm": round(miss * 100, 1) if miss is not None else None,
            "pad_speed": round(math.hypot(vx, vy), 3), "sam_calls": d.sam_calls}


results = []
for i in range(N):
    c = condition(i)
    row = dict(c)
    for m in METHODS:
        row[m] = run(c, m)
    results.append(row)
    print(f"#{i:2d} gust {c['gust']:.1f} wind {c['wind']} pad speed {row['rl']['pad_speed']:.2f} m/s | "
          + " | ".join(f"{m}: {'OK ' if row[m]['landed'] else 'MISS'} {row[m]['cm']} cm" for m in METHODS))

json.dump(results, open("exp_rl_landing.json", "w"), indent=2)
print("\n=========== SUMMARY (PyBullet, never seen in training) ===========")
print("detector:", "SAM 3.1" if USE_SAM else "simulator mask (SAM stand-in)")
for g in GUSTS + ["all"]:
    rows = [r for r in results if g == "all" or r["gust"] == g]
    if not rows:
        continue
    parts = []
    for m in METHODS:
        ok = [r[m] for r in rows if r[m]["landed"]]
        cm = np.mean([r["cm"] for r in ok]) if ok else float("nan")
        parts.append(f"{m} {len(ok)}/{len(rows)} ({cm:.1f} cm)")
    label = f"gusts {g:.1f}" if g != "all" else "ALL     "
    print(f"{label}: " + "  |  ".join(parts))
