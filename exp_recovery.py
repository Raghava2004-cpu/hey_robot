"""Does checking every step make the robots reliable when the real world goes wrong?
Hidden faults: each grasp slips with 30% chance, and in half the runs a worker moves an ordered item
to another station right after the drone has scanned it.
  naive    = the plan trusts the first answer (no retries, no re-scan)
  verified = the plan checks every result: retries slipped grasps, re-scans when an item is gone

python exp_recovery.py              # 20 orders x 2 plans, simulator mask instead of SAM (fast)
python exp_recovery.py --sam --n 4  # spot check with real SAM 3.1 + Spark checks (slow, API calls)
"""
import json
import os
import random
import sys

import pybullet as p

from mission_tasks import LAYOUT
from skill_tasks import ITEMS, deliver_check, fault_scene, plan_deliver, plan_deliver_verified
from team_skills import MAP_FILE, TeamAPI

ARGS = sys.argv[1:]
USE_SAM = "--sam" in ARGS
N = int(ARGS[ARGS.index("--n") + 1]) if "--n" in ARGS else 20


def run(seed, plan):
    rng = random.Random(seed)
    items = rng.sample([i for i in ITEMS if i in LAYOUT], 2)     # the first warehouse's 5 products
    if os.path.exists(MAP_FILE):
        os.remove(MAP_FILE)
    env = fault_scene(items, slip=0.3, move=(seed % 2 == 1), seed=seed)(gui=False)
    if not USE_SAM:
        import sim_vision
        sim_vision.install(env)
    api = TeamAPI(env)
    err = ""
    try:
        (plan_deliver_verified if plan == "verified" else plan_deliver)(api, items)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    env.step(240)
    ok, msg = deliver_check(items)(env)
    slips = sum(1 for s in api.log if s["action"] == "pick" and "slipped" in str(s["result"]))
    out = {"items": items, "ok": bool(ok), "msg": msg + (f" CRASH {err}" if err else ""), "slips": slips,
           "moved": list(env.fault_log), "actions": len(api.log)}
    p.disconnect()
    return out


rows = []
for seed in range(N):
    row = {"seed": seed, "naive": run(seed, "naive"), "verified": run(seed, "verified")}
    rows.append(row)
    v = row["verified"]
    print(f"#{seed:2d} {v['items']} moved={bool(v['moved'])} | naive: {'OK ' if row['naive']['ok'] else 'FAIL'} "
          f"(slips {row['naive']['slips']}) | verified: {'OK ' if v['ok'] else 'FAIL'} "
          f"(slips {v['slips']}, actions {v['actions']})")
json.dump(rows, open("exp_recovery.json", "w"), indent=2)

print("\n=========== SUMMARY ===========")
print("perception:", "SAM 3.1 + Spark" if USE_SAM else "simulator mask (SAM stand-in)")
for label, sel in [("no worker", lambda r: not r["verified"]["moved"]), ("worker moved an item", lambda r: bool(r["verified"]["moved"])),
                   ("ALL", lambda r: True)]:
    rs = [r for r in rows if sel(r)]
    if rs:
        print(f"{label:22s}: naive {sum(r['naive']['ok'] for r in rs)}/{len(rs)}   |   "
              f"verified {sum(r['verified']['ok'] for r in rs)}/{len(rs)}")
