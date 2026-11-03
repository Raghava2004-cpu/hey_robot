import pybullet as p

from drone_skills import DroneAPI
from landing_tasks import landing_scene

KP = 6.0
env = landing_scene()(gui=True)
api = DroneAPI(env)

# 1) hidden wind from hover drift
target = (-1.0, 1.0, 1.3)
pos = api.fly_to(*target)
dx, dy = pos[0] - target[0], pos[1] - target[1]
print(f"wind estimate {KP * dx:+.2f}, {KP * dy:+.2f} m/s^2 (true 0.80, 0.50)")
api.set_wind_correction(-KP * dx, -KP * dy)

# 2) rough robot velocity from its odometry, then fly over the pad
g1 = api.ground_robot_position()
api.wait(1.0)
g2 = api.ground_robot_position()
vx, vy = g2[0] - g1[0], g2[1] - g1[1]
api.fly_to(g2[0] - 0.38 + vx * 3, g2[1] + vy * 3, 1.3)

# 3) precise pad velocity from two camera sightings 1 s apart
p1 = api.find_pad()
api.wait(1.0)
p2 = api.find_pad()
print("pad seen:", p1, p2)
if p1 and p2:
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    # 4) rendezvous: where the pad will be after 3.0 s of flying + 2.0 s of descent
    rx, ry = p2[0] + vx * 5.0, p2[1] + vy * 5.0
    api.fly_to(rx, ry, 1.3)
    print("LANDED:", api.land_at(rx, ry))
    print(f"miss at touchdown: {env.drone.touchdown_miss * 100:.1f} cm")

for s in api.log:
    print("  ", s)
while p.isConnected():
    env.step()