import json
import os
import time

from PIL import Image

from imagine import imagine
from llm import spark, extract_code, spark_compare
from perception import _to_data_url
from skills import API_DOC, API_METHODS, RobotAPI, check_code, run_skill

LIBRARY = "skills_library.json"
DIAGNOSIS_MODEL = "muse-wonder-1.0"  # senior expert, used only after a failure
os.makedirs("runs", exist_ok=True)


def make_system(api_doc):
    return f"""You control a robot through a Python object `robot`.
Write ONE function `def skill(robot):` that completes the task.
Available methods:{api_doc}
Rules: no imports, no other functions or classes, only these methods and basic Python.
Keep the code short and simple (under 40 lines). Robot methods return False on failure instead of raising errors, so do NOT use try/except.
Check return values. Return True if the task is done, else False.
Reply with only one ```python code block."""


def load_library(path=LIBRARY):
    return json.load(open(path)) if os.path.exists(path) else []


def memory_text(lib):
    if not lib:
        return ""
    parts = ["\n\nMEMORY from past tasks, possibly on a different robot (reuse the ideas that help):"]
    for e in lib[-3:]:
        parts.append(f"- Task: {e['task']}\n  LESSON: {e.get('lesson', '')} "
                     f"(verified by: {e.get('verified_by', 'unknown')})\n"
                     f"  Proven skill:\n```python\n{e['code']}\n```")
    return "\n".join(parts)


def run_attempt(code, build_scene, check, gui, api_cls, methods, views=()):
    env = build_scene(gui)
    before, _ = env.camera(env.main_view)
    api = api_cls(env)
    problem = check_code(code, methods)
    if problem:
        result = f"rejected by safety check: {problem}"
    else:
        try:
            result = run_skill(code, api)
        except Exception as e:
            result = f"crashed: {e}"
    env.go_home()  # move the arm out of the camera's view
    after, _ = env.camera(env.main_view)
    extra_after = {}
    for v in views:                      # the other cameras of the goal checks (close-ups of tables)
        try:
            extra_after[v], _ = env.camera(v)
        except Exception:
            pass
    ok, msg = check(env)
    env.close()
    return ok, msg, result, api.log, before, after, extra_after


def learn(task, build_scene, check, use_memory=True, save=True, max_attempts=3,
          gui=True, verbose=True, use_imagination=True, library=LIBRARY,
          api_cls=RobotAPI, api_doc=API_DOC, methods=API_METHODS, imagine_text=None, extra_goals=()):
    """extra_goals = [(camera view, goal text), ...]: Muse Image also imagines the goal from these
    cameras (e.g. a close-up of the destination table) and Spark checks each one after the attempt."""
    lib = load_library(library) if use_memory else []
    goal_text = imagine_text or task
    extras = []                          # (view, text, imagined picture)

    # 1) IMAGINE the goal before acting (Muse Image)
    imagined = None
    if use_imagination:
        try:
            env0 = build_scene(False)
            start, _ = env0.camera(env0.main_view)
            starts = {}
            for v, _ in extra_goals:
                try:
                    starts[v], _ = env0.camera(v)
                except Exception:
                    pass
            env0.close()
            imagined = imagine(goal_text, start)
            Image.fromarray(imagined).save("runs/imagined.png")
            if verbose:
                print("🎨 IMAGINED the goal -> runs/imagined.png")
            for k, (v, txt) in enumerate(extra_goals, 1):
                if v not in starts:
                    continue
                try:
                    img = imagine(txt, starts[v])
                    Image.fromarray(img).save(f"runs/imagined_{k}.png")
                    extras.append((v, txt, img))
                    if verbose:
                        print(f"🎨 IMAGINED camera '{v}' -> runs/imagined_{k}.png")
                except Exception as e:
                    if verbose:
                        print(f"(imagination for '{v}' skipped:", str(e)[:80], ")")
        except Exception as e:
            if verbose:
                print("(imagination skipped:", str(e)[:100], ")")

    messages = [{"role": "system", "content": make_system(api_doc) + memory_text(lib)},
                {"role": "user", "content": task}]
    reply, _ = spark(messages, effort="low")
    imagination_log = []

    for attempt in range(1, max_attempts + 1):
        code = extract_code(reply)
        try:
            import muse_eye
            muse_eye.set_code(code)                     # the Muse Brain panel shows this program live
        except Exception:
            pass
        if verbose:
            print(f"\n--- attempt {attempt} code ---\n{code}")
        ok, msg, result, log, before, after, extra_after = run_attempt(
            code, build_scene, check, gui, api_cls, methods, views=[v for v, _, _ in extras])
        Image.fromarray(after).save(f"runs/after_{attempt}.png")
        try:
            import muse_eye
            muse_eye.event("CHECK", f"attempt {attempt}: {'PASS' if ok else 'FAIL'} - {msg}")
        except Exception:
            pass
        if verbose:
            print(f"--- attempt {attempt} --- returned={result}  CHECK: {msg}")
            for s in log:
                print("   ", s)

        # 2) Compare REALITY with IMAGINATION (Muse Spark vision)
        imag_note = ""
        if imagined is not None:
            try:
                match, why = spark_compare(goal_text, imagined, after)
                imagination_log.append({"attempt": attempt, "match": match, "checker": ok})
                imag_note = f"Imagination check: {'MATCH' if match else 'MISMATCH'} - {why}"
                if verbose:
                    print("👁️ ", imag_note)
                try:
                    import muse_eye
                    muse_eye.event("CHECK", f"Spark vs Muse Image goal: {'MATCH' if match else 'MISMATCH'}")
                except Exception:
                    pass
            except Exception as e:
                if verbose:
                    print("(imagination check skipped:", str(e)[:80], ")")
        for k, (v, txt, img) in enumerate(extras, 1):       # one more check per extra camera
            if v not in extra_after:
                continue
            try:
                Image.fromarray(extra_after[v]).save(f"runs/after_{attempt}_{k}.png")
                match, why = spark_compare(txt, img, extra_after[v])
                imagination_log.append({"attempt": attempt, "view": v, "match": match, "checker": ok})
                line = f"Imagination check ({v}): {'MATCH' if match else 'MISMATCH'} - {why}"
                imag_note += ("\n" if imag_note else "") + line
                if verbose:
                    print("👁️ ", line)
                try:
                    import muse_eye
                    muse_eye.event("CHECK", f"close-up vs goal: {'MATCH' if match else 'MISMATCH'}")
                except Exception:
                    pass
            except Exception as e:
                if verbose:
                    print(f"(imagination check for '{v}' skipped:", str(e)[:80], ")")

        if ok:
            lesson = ""
            if save:
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        "It worked. Write ONE short general LESSON (one sentence) that would "
                        "help with similar future tasks in other layouts, with other objects, "
                        "or on other robots. If you discovered a numeric correction, include "
                        "the number. Start with 'LESSON:'."},
                ]
                r, _ = spark(messages, effort="low")
                lesson = r.split("LESSON:", 1)[-1].strip().splitlines()[0]
                all_lib = load_library(library)
                all_lib.append({"task": task, "code": code, "attempts": attempt, "lesson": lesson,
                                "verified_by": "independent simulator checker",
                                "created": time.strftime("%Y-%m-%d %H:%M")})
                json.dump(all_lib, open(library, "w"), indent=2)
                if verbose:
                    print("💾 LESSON:", lesson)
            return {"success": True, "attempts": attempt, "code": code, "lesson": lesson,
                    "actions": len(log), "imagination": imagination_log}

        if attempt == max_attempts:      # no attempts left: skip the diagnosis
            break

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
        for k, (v, txt, img) in enumerate(extras, 1):
            if v in extra_after:
                content[0]["text"] += f"\nNext two images: camera '{v}' after the attempt, then its imagined goal."
                content.append({"type": "image_url", "image_url": {"url": _to_data_url(extra_after[v])}})
                content.append({"type": "image_url", "image_url": {"url": _to_data_url(img)}})
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
        try:
            import muse_eye
            muse_eye.event(who, reply.split("```")[0].strip().replace("DIAGNOSIS:", "").strip()[:68])
        except Exception:
            pass

    return {"success": False, "attempts": max_attempts, "code": None, "lesson": "",
            "actions": 0, "imagination": imagination_log}