from robot_env import RobotEnv

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)

print(env.pick("red"))
print(env.place(env.get_pos("tray")[:2]))
env.step(240)

red, tray = env.get_pos("red"), env.get_pos("tray")
inside = abs(red[0] - tray[0]) < 0.1 and abs(red[1] - tray[1]) < 0.1
print("Red cube in tray:", inside)

while True:
    env.step()