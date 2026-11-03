import json
import math
import os
import random
import shutil

from drone_skills import DRONE_DOC, DRONE_METHODS, DroneAPI
from landing_tasks import RESULT, TEXT, landed_check, landing_scene
from learner import learn
from summarize import wilson

FILE = "exp_landing_results.json"
MEM = "frozen_landing_memory.json"      # lessons from training landings only
if not os.path.exists(MEM):
    shutil.copy("landing_library.json", MEM)
CONDS = ["naive", "no_memory", "with_memory"]


def conditions(n=15):
    """15 unseen conditions: random wind (strength + direction), robot speed, drone start."""
    rng = random.Random(2026)
    out = []
    for i in range(n):
        mag, ang = rng.uniform(0.3, 1.3), rng.uniform(0, 2 * math.pi)
        out.append({"id": i,
                    "wind": [round(mag * math.cos(ang), 3), round(mag * math.sin(ang), 3)],
                    "speed": round(rng.uniform(0.9, 1.8), 2),
                    "start": [round(rng.uniform(-1.5, -0.5), 2), round(rng.uniform(-1.0, 1.0), 2), 1.2]})
    return out


def scene(c):
    return landing_scene(wind=tuple(c["wind"]), wheel_speed=c["speed"], drone_start=tuple(c["start"]))


def wind_error(c):
    ff, w = RESULT.get("ff"), c["wind"]
    if not ff or (ff[0] == 0 and ff[1] == 0):
        return None
    return round(100 * math.hypot(ff[0] + w[0], ff[1] + w[1]) / math.hypot(*w), 1)


def run_naive(c):
    """Physics-unaware baseline: fly over the pad and land where it is now."""
    env = scene(c)(False)
    api = DroneAPI(env)
    g = api.ground_robot_position()
    api.fly_to(g[0] - 0.38, g[1], 1.3)
    pad = api.find_pad()
    if pad:
        api.land_at(pad[0], pad[1])
    ok, _ = landed_check(env)
    env.close()
    return {"success": ok}


def run_skillforge(c, use_memory):
    r = learn(TEXT, scene(c), landed_check, use_memory=use_memory, save=False, max_attempts=1,
              gui=False, verbose=False, use_imagination=False, library=MEM,
              api_cls=DroneAPI, api_doc=DRONE_DOC, methods=DRONE_METHODS)
    return {"success": r["success"], "wind_error_pct": wind_error(c)}


results = json.load(open(FILE)) if os.path.exists(FILE) else {k: [] for k in CONDS}
for k in CONDS:
    results[k] = [r for r in results.get(k, []) if "error" not in r]
done = {(k, r["id"]) for k in CONDS for r in results[k]}

for c in conditions():
    for k in CONDS:
        if (k, c["id"]) in done:
            continue
        try:
            r = run_naive(c) if k == "naive" else run_skillforge(c, use_memory=(k == "with_memory"))
            r["miss_cm"] = round(RESULT["miss"] * 100, 1) if RESULT.get("miss") is not None else None
        except Exception as e:
            r = {"success": False, "error": str(e)}
        r.update(id=c["id"], wind=c["wind"], speed=c["speed"])
        results[k].append(r)
        print(f"cond {c['id']:2d}  {k:12s}  success={r['success']}  miss={r.get('miss_cm')} cm  "
              f"wind_err={r.get('wind_error_pct')}%  {r.get('error', '')}")
        json.dump(results, open(FILE, "w"), indent=2)

print("\n=========== LANDING EXPERIMENT SUMMARY (15 unseen conditions) ===========")
for k in CONDS:
    rs = results[k]
    n, wins = len(rs), sum(r["success"] for r in rs)
    lo, hi = wilson(wins, n)
    misses = [r["miss_cm"] for r in rs if r.get("success") and r.get("miss_cm") is not None]
    werrs = [r["wind_error_pct"] for r in rs if r.get("wind_error_pct") is not None]
    print(f"{k:12s}: landed {wins}/{n} [{lo:.0%}-{hi:.0%}]"
          + (f"  mean touchdown error {sum(misses) / len(misses):.1f} cm" if misses else "")
          + (f"  mean wind-estimate error {sum(werrs) / len(werrs):.1f}%" if werrs else ""))