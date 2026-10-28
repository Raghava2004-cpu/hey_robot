from perception import sam_find
from robot_env import RobotEnv

env = RobotEnv(gui=False)  # no window needed
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)
rgb, depth = env.camera()
for phrase in ["green tray", "green box", "tray", "green container"]:
    boxes = sam_find(rgb, phrase)
    confs = [round(b["conf"], 2) for b in boxes]
    print(f"{phrase:12s} -> {len(boxes)} match(es)  conf={confs}")