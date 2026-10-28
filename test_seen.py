from robot_env import RobotEnv

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)

for cube in ["red cube", "blue cube"]:
    print(cube, "->", env.pick_seen(cube))
    print("tray  ->", env.place_on("tray"))
    env.step(240)

# check with the simulator's true positions (for scoring only)
tray = env.get_pos("tray")
for name in ["red", "blue"]:
    pos = env.get_pos(name)
    inside = abs(pos[0] - tray[0]) < 0.1 and abs(pos[1] - tray[1]) < 0.1
    print(f"{name} in tray: {inside}")