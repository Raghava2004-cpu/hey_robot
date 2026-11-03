import os

import pybullet as p

from audit_tasks import AuditAPI, at_station, audit_check, audit_scene
from team_skills import MAP_FILE

if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)
env = audit_scene()(gui=True)
api = AuditAPI(env)
record = api.inventory_record()
print("RECORD :", record)

# 1) the drone audits every station (far look -> closer look -> Spark names the product)
for place in [pl for pl in api.places() if pl != "packing station"]:
    print(place, "->", api.drone_scan(place, list(record)))
seen = api.known_map()
print("DRONE SAW:", seen)
print("SPARK CHECKS:")
for c in api.checks:
    print("   ", c)

# 2) compare with the records: fix misplaced items, report missing ones
for item, should in record.items():
    where = seen.get(item)
    if where is None:
        api.report(f"missing: {item} (records say {should})")
        print(f"missing: {item}")
    elif where != should:
        print(f"misplaced: {item} is at {where}, should be at {should}")
        api.go_to(where)
        print("  pick:", api.pick(item))
        api.go_to(should)
        print("  put down:", api.put_aside())
        env.step(240)
        pos = [round(v, 3) for v in env.get_pos(item)]
        print(f"  {item} is now at {pos}; on {should}'s table: {at_station(env, item, should)}")

# 3) drive off; the drone measures the wind, finds the pad and lands with tracking
api.start_driving()
g = api.ground_robot_position()
pos = api.drone_fly_to(g[0], g[1], 1.3)
api.drone_set_wind_correction(-6 * (pos[0] - g[0]), -6 * (pos[1] - g[1]))
p1 = api.drone_find_pad()
g = api.ground_robot_position()
api.drone_fly_to(g[0] - 0.38 + 0.6, g[1], 1.3)
p2 = api.drone_find_pad()
print("pad:", p1, p2)
vx, vy = ((p2[0] - p1[0]) / 3.0, (p2[1] - p1[1]) / 3.0) if p1 and p2 else (0.2, 0.0)
print("LANDED:", api.drone_land_tracking(vx, vy))
print("clean pad sightings [t, x, y]:", env.drone.track_log)
print("fitted pad velocity:", env.drone.track_velocity)
if env.drone.touchdown_miss is not None:
    print(f"touchdown error: {env.drone.touchdown_miss * 100:.1f} cm")

print("CHECK:", audit_check(env))
for s in api.log:
    print("  ", s)
while p.isConnected():
    env.step()