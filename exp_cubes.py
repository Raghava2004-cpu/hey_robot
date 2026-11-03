import json
import os
import time

from agent_policy import act_live
from cube_skills import CUBE_DOC, CUBE_METHODS, CubeAPI
from cube_tasks import STACK_TASK, cube_scene, spell_check, spell_task, stack_check
from learner import learn
from summarize import wilson

FILE = "exp_cubes_results.json"
SEEDS = range(1, 6)                     # 5 unseen layouts (5 independent lineages)
WORD = "META"
CONDS = ["lineage", "vanilla", "best_of_3", "live_agent"]
API = dict(api_cls=CubeAPI, api_doc=CUBE_DOC, methods=CUBE_METHODS)
KW = dict(gui=False, verbose=False, use_imagination=False, **API)


def reuse(new, old):
    """Share of the spelling skill's code lines that already existed in the stacking skill."""
    def lines(c):
        return {l.strip() for l in (c or "").splitlines() if len(l.strip()) > 8 and not l.strip().startswith("#")}
    a, b = lines(new), lines(old)
    return round(len(a & b) / len(a), 2) if a and b else None


def run(cond, seed):
    t0 = time.time()
    task, scene, check = spell_task(WORD), cube_scene(seed), spell_check(WORD)
    if cond == "lineage":                       # stack first (new layout), then spell with that library
        lib = f"lineage_{seed}.json"
        if os.path.exists(lib):
            os.remove(lib)
        s = learn(STACK_TASK, cube_scene(seed + 100), stack_check(4), library=lib, save=True, **KW)
        r = learn(task, scene, check, library=lib, save=True, **KW)
        out = {"success": r["success"], "attempts": r["attempts"], "stack_success": s["success"],
               "stack_attempts": s["attempts"], "code_reuse": reuse(r["code"], s["code"])}
    elif cond == "vanilla":                     # same task, empty library, 3 tries with Wonder diagnosis
        r = learn(task, scene, check, use_memory=False, save=False, **KW)
        out = {"success": r["success"], "attempts": r["attempts"]}
    elif cond == "best_of_3":                   # 3 fresh tries, no library, no diagnosis
        out = {"success": False, "attempts": 3}
        for k in range(1, 4):
            r = learn(task, scene, check, use_memory=False, save=False, max_attempts=1, **KW)
            if r["success"]:
                out = {"success": True, "attempts": k}
                break
    else:                                       # Spark acts live, one action at a time
        r = act_live(task, scene, check, gui=False, verbose=False, **API)
        out = {"success": r["success"], "attempts": 1, "steps": r["steps"], "model_calls": r["model_calls"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def pct(k, n):
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.0%}-{hi:.0%}]"


results = json.load(open(FILE)) if os.path.exists(FILE) else {}
for c in CONDS:
    results[c] = [r for r in results.get(c, []) if "error" not in r]
done = {(c, r["seed"]) for c in CONDS for r in results[c]}

for seed in SEEDS:
    for c in CONDS:
        if (c, seed) in done:
            continue
        try:
            r = run(c, seed)
        except Exception as e:
            r = {"success": False, "attempts": 0, "error": str(e)}
        r["seed"] = seed
        results[c].append(r)
        print(f"seed {seed}  {c:11s}  {r}")
        json.dump(results, open(FILE, "w"), indent=2)

print(f"\n=========== SPELL {WORD}: {len(list(SEEDS))} unseen layouts ===========")
for c in CONDS:
    rs = results[c]
    n = len(rs)
    wins = sum(r["success"] for r in rs)
    first = sum(r["success"] and r["attempts"] == 1 for r in rs)
    secs = [r["seconds"] for r in rs if "seconds" in r]
    line = f"{c:11s}: success {pct(wins, n):16s} first-try {pct(first, n):16s}"
    line += f" mean time {sum(secs) / len(secs):.0f} s" if secs else ""
    if c == "live_agent":
        calls = [r["model_calls"] for r in rs if "model_calls" in r]
        line += f"  mean model calls {sum(calls) / len(calls):.1f}" if calls else ""
    if c == "lineage":
        st = sum(r.get("stack_success", False) for r in rs)
        ru = [r["code_reuse"] for r in rs if r.get("code_reuse") is not None]
        line += f"  stack success {st}/{n}"
        line += f"  code reused from stacking {sum(ru) / len(ru):.0%}" if ru else ""
    print(line)
print("\nSEAR reading: lineage vs vanilla = library effect; vanilla vs best_of_3 = learning vs retrying;"
      "\nlive_agent vs vanilla = acting live vs writing code (compare time and model calls too).")