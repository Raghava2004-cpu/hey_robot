import json
import os

from learner import learn
from tasks import random_hidden_task
from summarize import summarize

FILE = "benchmark_results.json"
SEEDS = range(1, 11)

results = json.load(open(FILE)) if os.path.exists(FILE) else {"no_memory": [], "with_memory": []}
for cond in results:
    results[cond] = [r for r in results[cond] if "error" not in r]  # drop network failures
done = {(cond, r["seed"]) for cond in results for r in results[cond]}

for seed in SEEDS:
    for cond in results:
        if (cond, seed) in done:
            continue
        task, build, check = random_hidden_task(seed)
        try:
            r = learn(task, build, check, use_memory=(cond == "with_memory"),
                      save=False, gui=False, verbose=False)
        except Exception as e:
            r = {"success": False, "attempts": 0, "actions": 0, "error": str(e)}
        r.pop("code", None)
        r.update(seed=seed, task=task)
        results[cond].append(r)
        results[cond].sort(key=lambda x: x["seed"])
        print(f"seed {seed:2d}  {cond:12s}  success={r['success']}  "
              f"attempts={r['attempts']}  actions={r['actions']}  {r.get('error', '')}")
        json.dump(results, open(FILE, "w"), indent=2)

summarize(results)