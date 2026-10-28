from llm import spark, extract_code
from robot_env import RobotEnv
from skills import API_DOC, RobotAPI, check_code, run_skill

SYSTEM = f"""You control a robot arm through a Python object `robot`.
Write ONE function `def skill(robot):` that completes the task.
Available methods:{API_DOC}
Rules: no imports, no other functions or classes, only these methods and basic Python.
Check return values. Return True if the task is done, else False.
Reply with only one ```python code block."""

TASK = "Put the red cube and the blue cube into the tray."

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)

reply, usage = spark([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": TASK}])
code = extract_code(reply)
print("=== Spark wrote ===\n" + code)
print("tokens:", usage.get("total_tokens"))

problem = check_code(code)
if problem:
    print("REJECTED by safety check:", problem)
else:
    api = RobotAPI(env)
    try:
        result = run_skill(code, api)
    except Exception as e:
        result = f"crashed: {e}"
    print("\nskill returned:", result)
    for step in api.log:
        print("  ", step)

env.step(240)
tray = env.get_pos("tray")
for name in ["red", "blue"]:
    pos = env.get_pos(name)
    print(f"{name} in tray:", abs(pos[0] - tray[0]) < 0.1 and abs(pos[1] - tray[1]) < 0.1)