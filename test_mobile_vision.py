import math

import pybullet as p
from PIL import Image

from mobile_env import MobileEnv

env = MobileEnv(gui=True)
top = env.add_table("pickup table", (1.6, 0.0))                        # in front of the robot
env.add_cube("red", (1.42, 0.08), top, [1, 0, 0, 1])
env.add_cube("blue", (1.42, -0.10), top, [0, 0, 1, 1])
pack = env.add_table("packing station", (0.0, -1.6), rgba=(0.6, 0.5, 0.4, 1))
env.add_tray("tray", (0.0, -1.42), pack)
for xy in [(3.0, 1.5), (3.0, -1.5)]:                                   # warehouse shelves
    shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
    p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
env.step(120)

print(f"drive to pickup table: error {env.navigate_to(0.75, 0.0) * 100:.1f} cm")
print(f"face the table: error {math.degrees(env.turn_to(0.0)):.1f} deg")

rgb, _ = env.camera()
Image.fromarray(rgb).save("mobile_cam.png")
print("saved mobile_cam.png (what the robot sees)")

for name in ["red", "blue"]:
    found = env.perceive(f"{name} cube")
    true = env.get_pos(name)
    if not found:
        print(f"{name} cube: NOT seen")
        continue
    est = found[0]["pos"]
    est[2] -= 0.025                           # camera sees the top face
    err = math.dist(est, true) * 100
    print(f"{name} cube: conf={found[0]['conf']:.2f}  3D error={err:.1f} cm")

while p.isConnected():
    env.step()