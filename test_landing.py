import math

import pybullet as p

from drone import Drone
from mobile_env import MobileEnv

env = MobileEnv(gui=True)
env.add_landing_pad()
drone = Drone(env, wind=(0.8, 0.5))        # hidden wind: the robot does not know it
env.step(240)

# 1) hover test: the wind pushes the drone off target
pos = drone.fly_to(0.0, 0.8, 1.2, 3)
print(f"hover drift: dx={pos[0] - 0.0:+.3f} m, dy={pos[1] - 0.8:+.3f} m  (target was 0.0, 0.8)")

# 2) find the landing pad with SAM 3.1 from above
drone.fly_to(-0.3, 0.0, 1.3, 3)
pad = drone.find("landing pad")
print("pad seen by SAM at:", pad)

from PIL import Image
rgb, _ = drone.camera_down()
Image.fromarray(rgb).save("drone_cam.png")

# 3) the Husky starts driving; the drone naively lands where the pad IS now
env.set_wheels(3, 3)                        # about 0.5 m/s forward
pad = drone.find("landing pad")
print("pad now at:", pad)
landed = drone.land_at(pad[0], pad[1]) if pad else False
env.set_wheels(0, 0)
env.step(120)

true_pad = p.getBasePositionAndOrientation(env.objects["landing pad"])[0]
d = drone.position()
print("landed on pad:", landed)
print(f"miss distance (true, for scoring only): {math.dist(d[:2], true_pad[:2]) * 100:.0f} cm")

while p.isConnected():
    env.step()