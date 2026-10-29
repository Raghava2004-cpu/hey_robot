import json
import os

from PIL import Image

from imagine import imagine
from llm import spark, extract_code, spark_compare
from perception import _to_data_url
from skills import API_DOC, RobotAPI, check_code, run_skill

LIBRARY = "skills_library.json"
DIAGNOSIS_MODEL = "muse-wonder-1.0"  # senior expert, used only after a failure
os.makedirs("runs", exist_ok=True)

BASE_SYSTEM = f"""You control a robot arm through a Python object `robot`.
Write ONE function `def skill(robot):` that completes the task.
Available methods:{API_DOC}
Rules: no imports, no other functions or classes, only these methods and basic Python.
Check return values. Return True if the task is done, else False.
Reply with only one ```python code block."""


def load_library():
    return json.load(open(LIBRARY)) if os.path.exists(LIBRARY) else []


def memory_text(lib):
    if not lib:
        return ""
    parts = ["\n\nMEMORY from past tasks (reuse what helps):"]
    for e in lib[-3:]:
        parts.append(f"- Task: {e['task']}\n  LESSON: {e.get('lesson', '')}\n"
                     f"  Proven skill:\n```python\n{e['code']}\n```")
    return "\n".join(parts)


def run_attempt(code, build_scene, check, gui):
    env = build_scene(gui)
    before, _ = env.camera()
    api = RobotAPI(env)
    problem = check_code(code)
    if problem:
        result = f"rejected by safety check: {problem}"
    else:
        try:
            result = run_skill(code, api)
        except Exception as e:
            result = f"crashed: {e}"
    env.go_home()  # move the arm out of the camera's view
    after, _ = env.camera()
    ok, msg = check(env)
    env.close()
    return ok, msg, result, api.log, before, after


def learn(task, build_scene, check, use_memory=True, save=True,
          max_attempts=3, gui=True, verbose=True, use_imagination=True):
    lib = load_library() if use_memory else []

    # 1) IMAGINE the goal before acting (Muse Image)
    imagined = None
    if use_imagination:
        try:
            env0 = build_scene(False)
            start, _ = env0.camera()
            env0.close()
            imagined = imagine(task, start)
            Image.fromarray(imagined).save("runs/imagined.png")
            if verbose:
                print("🎨 IMAGINED the goal -> runs/imagined.png")
        except Exception as e:
            if verbose:
                print("(imagination skipped:", str(e)[:100], ")")

    messages = [{"role": "system", "content": BASE_SYSTEM + memory_text(lib)},
                {"role": "user", "content": task}]
    reply, _ = spark(messages, effort="low")
    imagination_log = []

    for attempt in range(1, max_attempts + 1):
        code = extract_code(reply)
        ok, msg, result, log, before, after = run_attempt(code, build_scene, check, gui)
        Image.fromarray(after).save(f"runs/after_{attempt}.png")
        if verbose:
            print(f"\n--- attempt {attempt} --- returned={result}  CHECK: {msg}")
            for s in log:
                print("   ", s)

        # 2) Compare REALITY with IMAGINATION (Muse Spark vision)
        imag_note = ""
        if imagined is not None:
            try:
                match, why = spark_compare(task, imagined, after)
                imagination_log.append({"attempt": attempt, "match": match, "checker": ok})
                imag_note = f"Imagination check: {'MATCH' if match else 'MISMATCH'} - {why}"
                if verbose:
                    print("👁️ ", imag_note)
            except Exception as e:
                if verbose:
                    print("(imagination check skipped:", str(e)[:80], ")")

        if ok:
            lesson = ""
            if save:
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        "It worked. Write ONE short general LESSON (one sentence) that would "
                        "help with similar future tasks in other layouts and with other objects. "
                        "Start with 'LESSON:'."},
                ]
                r, _ = spark(messages, effort="low")
                lesson = r.split("LESSON:", 1)[-1].strip().splitlines()[0]
                all_lib = load_library()
                all_lib.append({"task": task, "code": code, "attempts": attempt, "lesson": lesson})
                json.dump(all_lib, open(LIBRARY, "w"), indent=2)
                if verbose:
                    print("💾 LESSON:", lesson)
            return {"success": True, "attempts": attempt, "code": code, "lesson": lesson,
                    "actions": len(log), "imagination": imagination_log}

        # 3) DIAGNOSE the failure (Muse Wonder), with the imagined goal as extra evidence
        feedback = (f"Attempt {attempt} FAILED.\nYour code returned: {result}\n"
                    f"Action log:\n{json.dumps(log, indent=1)}\nChecker: {msg}\n"
                    + (imag_note + "\n" if imag_note else "")
                    + "Image 1 = camera view before the attempt. Image 2 = after."
                    + (" Image 3 = the imagined goal." if imagined is not None else "")
                    + "\nFirst write 'DIAGNOSIS:' with 1-2 sentences on what went wrong and why. "
                      "Then write the fixed skill in one ```python block.")
        content = [{"type": "text", "text": feedback},
                   {"type": "image_url", "image_url": {"url": _to_data_url(before)}},
                   {"type": "image_url", "image_url": {"url": _to_data_url(after)}}]
        if imagined is not None:
            content.append({"type": "image_url", "image_url": {"url": _to_data_url(imagined)}})
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": content}]
        try:
            reply, _ = spark(messages, effort="medium", model=DIAGNOSIS_MODEL)
            who = "WONDER"
        except Exception as e:
            if verbose:
                print("(Wonder unavailable, using Spark:", str(e)[:80], ")")
            reply, _ = spark(messages, effort="medium")
            who = "SPARK"
        if verbose:
            print(f"🧠 {who}", reply.split("```")[0].strip())

    return {"success": False, "attempts": max_attempts, "code": None, "lesson": "",
            "actions": 0, "imagination": imagination_log}