import math

import pybullet as p

from mobile_env import MobileEnv

env = MobileEnv(gui=True)
top = env.add_table("pickup table", (1.6, 0.0))
env.add_cube("red", (1.42, 0.08), top, [1, 0, 0, 1])
env.add_cube("blue", (1.42, -0.10), top, [0, 0, 1, 1])
pack = env.add_table("packing station", (0.0, -1.6), rgba=(0.6, 0.5, 0.4, 1))
env.add_tray("tray", (0.0, -1.42), pack)
for xy in [(3.0, 1.5), (3.0, -1.5)]:
    shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
    p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
env.step(120)

# 1) go to the pickup table and pick the red cube (seen by SAM 3.1)
env.navigate_to(0.75, 0.0)
env.turn_to(0.0)
print("pick:", env.pick_seen("red cube"))

# 2) drive to the packing station and place it in the tray
print(f"drive to packing station: error {env.navigate_to(0.0, -0.70) * 100:.1f} cm")
env.turn_to(-math.pi / 2)
print("place:", env.place_on("tray"))
env.step(240)

tray, red = env.get_pos("tray"), env.get_pos("red")
print("red cube in tray:", abs(red[0] - tray[0]) < 0.1 and abs(red[1] - tray[1]) < 0.1)

while p.isConnected():
    env.step()