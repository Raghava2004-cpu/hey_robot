import pybullet as p
from robot_env import RobotEnv

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.step(120)


def sabotage():
    x, y, z = env.get_pos("red")
    p.resetBasePositionAndOrientation(env.objects["red"], [x + 0.06, y + 0.05, z], [0, 0, 0, 1])
    print(">>> SABOTAGE: red cube moved ~8 cm while the arm was reaching!")


env.hover_hook = sabotage
print("pick:", env.pick_seen("red cube"))
env.hover_hook = None
print("tray:", env.place_on("tray"))
env.step(240)

tray, red = env.get_pos("tray"), env.get_pos("red")
print("red in tray:", abs(red[0] - tray[0]) < 0.1 and abs(red[1] - tray[1]) < 0.1)