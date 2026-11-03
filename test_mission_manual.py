import os

import pybullet as p

from mission_tasks import mission_scene
from team_skills import MAP_FILE, TeamAPI

if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)
env = mission_scene()(gui=True)
api = TeamAPI(env)
items = ["banana", "soup can"]

# 1) drone scouts
for place in ["station A", "station B", "station C"]:
    print(place, "->", api.drone_scan(place, items))
print("map:", api.known_map())

# 2) ground robot fetches from the map
for it in items:
    place = api.recall(it)
    print(f"fetch {it} from {place}")
    api.go_to(place)
    print("  pick:", api.pick(it))
    api.go_to("packing station")
    print("  place:", api.place_on("tray"))

# 3) drive off, drone lands (rendezvous math)
api.start_driving()
gx, gy = api.ground_robot_position()
pos = api.drone_fly_to(gx, gy, 1.3)
api.drone_set_wind_correction(-6 * (pos[0] - gx), -6 * (pos[1] - gy))
p1 = api.drone_find_pad()
g = api.ground_robot_position()
pos = api.drone_fly_to(g[0] - 0.38 + 0.6, g[1], 1.3)
p2 = api.drone_find_pad()
print("pad:", p1, p2)
if p1 and p2:
    vx, vy = (p2[0] - p1[0]) / 3.0, (p2[1] - p1[1]) / 3.0
    rx, ry = p2[0] + vx * 5.0, p2[1] + vy * 5.0
    api.drone_fly_to(rx, ry, 1.3)
    print("LANDED:", api.drone_land_at(rx, ry))

from mission_tasks import mission_check
print("CHECK:", mission_check(items)(env))
for s in api.log:
    print("  ", s)
while p.isConnected():
    env.step()