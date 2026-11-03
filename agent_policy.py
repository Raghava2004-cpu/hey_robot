import time

from learner import load_library, memory_text
from llm import spark
from perception import _to_data_url
from skills import check_code, run_skill

LIVE_SYSTEM = """You control a robot LIVE, one action at a time, through a Python object `robot`.
Available methods:{doc}
Each turn you get the task, everything you did so far with the results, and the current camera image.
Reply with EXACTLY ONE line:
- one robot call with plain numbers, e.g.  robot.look()   or   robot.pick(0.74, -0.12, 0.5)
- or DONE   when the task is complete
- or GIVE UP   if it is impossible.
Use numbers you have seen in earlier results. No other text."""


def _parse(reply):
    for line in reply.replace("```python", "\n").replace("```", "\n").splitlines():
        s = line.strip()
        if s.startswith("robot."):
            return s
        if s.upper().startswith("DONE"):
            return "DONE"
        if s.upper().startswith("GIVE UP"):
            return "GIVE UP"
    return None


def act_live(task, build_scene, check, api_cls, api_doc, methods, max_steps=30,
             gui=True, verbose=True, use_memory=False, library=None):
    t0 = time.time()
    env = build_scene(gui)
    api = api_cls(env)
    system = LIVE_SYSTEM.format(doc=api_doc)
    if use_memory and library:
        system += memory_text(load_library(library))
    history, calls = [], 0
    for step in range(1, max_steps + 1):
        rgb, _ = env.camera(env.main_view)
        text = (f"TASK: {task}\n\nSO FAR:\n" + ("\n".join(history) if history else "(nothing yet)")
                + f"\n\nStep {step} of {max_steps}. Your one line:")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": _to_data_url(rgb)}}]}]
        reply, _ = spark(messages, effort="low")
        calls += 1
        line = _parse(reply)
        if line is None:
            history.append(f"{step}. (invalid reply: answer with one robot call, DONE or GIVE UP)")
            continue
        if line in ("DONE", "GIVE UP"):
            history.append(f"{step}. {line}")
            if verbose:
                print(f"  [{step}] {line}")
            break
        code = f"def skill(robot):\n    return {line}\n"
        problem = check_code(code, methods)
        if problem:
            result = f"REFUSED by safety check: {problem}"
        else:
            try:
                result = run_skill(code, api)
            except Exception as e:
                result = f"error: {e}"
        r = str(result)
        if len(r) > 600:
            r = r[:600] + "..."
        history.append(f"{step}. {line} -> {r}")
        if verbose:
            print(f"  [{step}] {line} -> {r[:160]}")
    env.go_home()
    ok, msg = check(env)
    env.close()
    return {"success": ok, "check": msg, "steps": len(history), "model_calls": calls,
            "seconds": round(time.time() - t0, 1), "history": history}