import json
import os
import shutil

from learner import learn
from summarize import wilson
from tasks import random_hidden_task

FILE = "benchmark20_results.json"
MEM = "frozen_arm_memory.json"          # memory from TRAINING tasks only, frozen for the test
if not os.path.exists(MEM):
    shutil.copy("skills_library.json", MEM)
SEEDS = range(1, 21)                    # 20 unseen random layouts, same for every condition
CONDS = ["no_memory", "with_memory", "best_of_3"]
KW = dict(save=False, gui=False, verbose=False, use_imagination=False, library=MEM)


def run(cond, task, build, check):
    if cond == "best_of_3":
        # SEAR control: 3 FRESH attempts, no memory, no diagnosis -> "just trying more often"
        for k in range(1, 4):
            r = learn(task, build, check, use_memory=False, max_attempts=1, **KW)
            if r["success"]:
                return {"success": True, "attempts": k, "actions": r["actions"]}
        return {"success": False, "attempts": 3, "actions": 0}
    r = learn(task, build, check, use_memory=(cond == "with_memory"), max_attempts=3, **KW)
    return {"success": r["success"], "attempts": r["attempts"], "actions": r["actions"]}


def pct(k, n):
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.0%}-{hi:.0%}]"


results = json.load(open(FILE)) if os.path.exists(FILE) else {}
for cond in CONDS:
    results[cond] = [r for r in results.get(cond, []) if "error" not in r]   # retry network failures
done = {(cond, r["seed"]) for cond in CONDS for r in results[cond]}

for seed in SEEDS:
    for cond in CONDS:
        if (cond, seed) in done:
            continue
        task, build, check = random_hidden_task(seed)
        try:
            r = run(cond, task, build, check)
        except Exception as e:
            r = {"success": False, "attempts": 0, "actions": 0, "error": str(e)}
        r.update(seed=seed, task=task)
        results[cond].append(r)
        results[cond].sort(key=lambda x: x["seed"])
        print(f"seed {seed:2d}  {cond:12s}  success={r['success']}  "
              f"attempts={r['attempts']}  actions={r['actions']}  {r.get('error', '')}")
        json.dump(results, open(FILE, "w"), indent=2)

print("\n=========== BENCHMARK: 20 unseen hidden-object layouts ===========")
for cond in CONDS:
    rs = results[cond]
    n = len(rs)
    wins = sum(r["success"] for r in rs)
    first = sum(r["success"] and r["attempts"] == 1 for r in rs)
    tries = [r["attempts"] for r in rs if r["success"]]
    avg = f"{sum(tries) / len(tries):.2f}" if tries else "-"
    print(f"{cond:12s}: success {pct(wins, n):18s} first-try {pct(first, n):18s} avg attempts {avg}")
print("\nRead it like SEAR:")
print("  memory effect      = with_memory vs no_memory (first-try column)")
print("  learning vs luck   = no_memory (3 tries WITH Wonder diagnosis) vs best_of_3 (3 fresh tries, NO diagnosis)")