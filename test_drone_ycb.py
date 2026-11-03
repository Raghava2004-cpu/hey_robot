import math
import os

import pybullet as p

from mobile_env import MobileEnv

YCB = os.path.join("third_party", "pybullet-object-models", "pybullet_object_models", "ycb_objects")
DRONE = os.path.join("third_party", "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")
print("YCB objects available:", sorted(os.listdir(YCB))[:30])

env = MobileEnv(gui=True)
info = env.add_station("station A", (1.6, 0.0), 0.0)
for name, (x, y) in zip(["YcbBanana", "YcbMustardBottle", "YcbTomatoSoupCan"], info["items"]):
    path = os.path.join(YCB, name, "model.urdf")
    if os.path.exists(path):
        env.objects[name] = p.loadURDF(path, [x, y, info["top"] + 0.1])
    else:
        print("missing:", name)

# --- drone: simplified flight physics (thrust + drag), controlled by a PD controller ---
drone = p.loadURDF(DRONE, [0.0, 0.0, 1.2], globalScaling=5)
p.changeDynamics(drone, -1, angularDamping=0.9)
mass = p.getDynamicsInfo(drone, -1)[0]


def fly_to(target, seconds):
    for _ in range(int(seconds * 240)):
        pos, _ = p.getBasePositionAndOrientation(drone)
        vel, _ = p.getBaseVelocity(drone)
        acc = [6.0 * (t - x) - 4.0 * v for t, x, v in zip(target, pos, vel)]
        force = [mass * acc[0], mass * acc[1], mass * (acc[2] + 9.81)]
        p.applyExternalForce(drone, -1, force, pos, p.WORLD_FRAME)
        env.step(1)
    pos, _ = p.getBasePositionAndOrientation(drone)
    return math.dist(pos, target)


env.step(120)
print(f"hover error: {fly_to([0.8, 0.6, 1.3], 3) * 100:.1f} cm")
print(f"over station A error: {fly_to([1.6, 0.0, 1.3], 3) * 100:.1f} cm")
hx, hy, _ = env.pose()
print(f"above Husky error: {fly_to([hx, hy, 1.0], 3) * 100:.1f} cm")
fly_to([hx, hy, 0.0], 3)   # descend onto the robot
print("landed on Husky:", len(p.getContactPoints(drone, env.husky)) > 0)

while p.isConnected():
    env.step()