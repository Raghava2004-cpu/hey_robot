import json
import os

from llm import spark, extract_code
from perception import _to_data_url
from robot_env import RobotEnv
from skills import API_DOC, RobotAPI, check_code, run_skill

SYSTEM = f"""You control a robot arm through a Python object `robot`.
Write ONE function `def skill(robot):` that completes the task.
Available methods:{API_DOC}
Rules: no imports, no other functions or classes, only these methods and basic Python.
Check return values. Return True if the task is done, else False.
Reply with only one ```python code block."""

TASK = "Put the blue cube into the tray."
MAX_ATTEMPTS = 3
LIBRARY = "skills_library.json"


def build_scene():
    env = RobotEnv(gui=True)
    env.add_cube("blue", [0.58, 0.12], [0, 0, 1, 1])
    env.step(60)
    env.add_cube_on("red", "blue", [1, 0, 0, 1])  # red hides blue from the camera
    env.step(120)
    return env


def check(env):
    tray, blue = env.get_pos("tray"), env.get_pos("blue")
    ok = abs(blue[0] - tray[0]) < 0.1 and abs(blue[1] - tray[1]) < 0.1
    return ok, "blue cube is in the tray" if ok else "blue cube is NOT in the tray"


messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
reply, _ = spark(messages, effort="low")

for attempt in range(1, MAX_ATTEMPTS + 1):
    code = extract_code(reply)
    print(f"\n========== ATTEMPT {attempt} ==========\n{code}")

    env = build_scene()
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
    env.step(240)
    after, _ = env.camera()
    ok, msg = check(env)
    env.close()

    print("returned:", result)
    for s in api.log:
        print("  ", s)
    print("CHECK:", msg)

    if ok:
        lib = json.load(open(LIBRARY)) if os.path.exists(LIBRARY) else []
        lib.append({"task": TASK, "code": code, "attempts": attempt})
        json.dump(lib, open(LIBRARY, "w"), indent=2)
        print(f"\n✅ Learned in {attempt} attempt(s). Saved to {LIBRARY}")
        break

    feedback = (f"Attempt {attempt} FAILED.\nYour code returned: {result}\n"
                f"Action log:\n{json.dumps(api.log, indent=1)}\nChecker: {msg}\n"
                "Image 1 = camera view before the attempt. Image 2 = after.\n"
                "First write 'DIAGNOSIS:' with 1-2 sentences on what went wrong and why. "
                "Then write the fixed skill in one ```python block.")
    messages += [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": [
            {"type": "text", "text": feedback},
            {"type": "image_url", "image_url": {"url": _to_data_url(before)}},
            {"type": "image_url", "image_url": {"url": _to_data_url(after)}},
        ]},
    ]
    reply, _ = spark(messages, effort="medium")
    print("\n🧠 SPARK", reply.split("```")[0].strip())
else:
    print("\n❌ Not learned within", MAX_ATTEMPTS, "attempts")