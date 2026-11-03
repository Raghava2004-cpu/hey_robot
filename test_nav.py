import math

import pybullet as p

from mobile_env import MobileEnv

env = MobileEnv(gui=True)
for xy in [(3.0, 1.5), (3.0, -1.5)]:          # two warehouse shelves for scenery
    shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
    p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])

print(f"to (2.0, 0.0): final error {env.navigate_to(2.0, 0.0) * 100:.1f} cm")
print(f"turn to 90 deg: final error {math.degrees(env.turn_to(math.pi / 2)):.1f} deg")
print(f"to (1.0, 1.0): final error {env.navigate_to(1.0, 1.0) * 100:.1f} cm")
print(f"back home (0, 0): final error {env.navigate_to(0.0, 0.0) * 100:.1f} cm")

while p.isConnected():
    env.step()