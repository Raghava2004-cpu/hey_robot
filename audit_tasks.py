import math
import os

import pybullet as p

from drone import Drone
from mission_tasks import STATIONS, YCB, YCB_FILES
YCB_FILES.setdefault("power drill", "YcbPowerDrill")   # importing it also registers item names for SAM
from mobile_env import MobileEnv
from team_skills import TEAM_DOC, TEAM_METHODS, TeamAPI

TABLE_TOP = 0.45
# what the warehouse management system SAYS
RECORD = {"banana": "station B", "soup can": "station C", "power drill": "station A"}
# what is REALLY on the tables: the soup can was put back at the wrong station, the drill is gone
ACTUAL = {"foam brick": ("station A", 0), "soup can": ("station A", 2),
          "banana": ("station B", 1), "gelatin box": ("station C", 2)}

AUDIT_METHODS = TEAM_METHODS | {"inventory_record", "report"}
AUDIT_DOC = TEAM_DOC + """
INVENTORY SYSTEM:
robot.inventory_record() -> dict {item: place}: where each item SHOULD be, according to the warehouse records (the real tables may not match)
robot.report(text) -> True   (send a short message to the warehouse manager, e.g. "missing: stapler (should be at station B)")
To move an item to another station: go_to its station, pick it, go_to the right station, then put_aside() puts it down on that station's table.
"""

AUDIT_TEXT = ("Inventory audit. Use the drone to check every station against the inventory records. "
              "If a recorded item is at the wrong station, the ground robot must move it back to the "
              "station where the records say it belongs. Report every recorded item that is missing "
              "from all stations. Then the ground robot drives away, and the drone must land on the "
              "landing pad on top of the moving ground robot. Wind may be blowing.")


class AuditAPI(TeamAPI):
    def inventory_record(self):
        return dict(self.env.record)

    def report(self, text):
        text = str(text)[:200]
        self.env.reports.append(text)
        self._record("report", text, "sent to the warehouse manager")
        return True


def audit_scene(wind=(0.8, 0.5), actual=None, record=None):
    actual = actual or ACTUAL
    record = record or RECORD

    def build(gui=False):
        env = MobileEnv(gui=gui)
        info = {n: env.add_station(n, c, yaw) for n, (c, yaw) in STATIONS.items()}
        env.station_centers = {n: c for n, (c, yaw) in STATIONS.items()}
        env.station_yaws = {n: yaw for n, (c, yaw) in STATIONS.items()}
        pk = env.add_station("packing station", (0.0, -1.6), -math.pi / 2, rgba=(0.45, 0.35, 0.25, 1))
        env.station_centers["packing station"] = (0.0, -1.6)
        env.station_yaws["packing station"] = -math.pi / 2
        env.add_tray("tray", (0.0, -1.45), pk["top"], size=0.3)
        for name, (st, k) in actual.items():
            x, y = info[st]["items"][k]
            env.objects[name] = p.loadURDF(os.path.join(YCB, YCB_FILES[name], "model.urdf"),
                                           [x, y, info[st]["top"] + 0.06])
            if name == "gelatin box":                     # square to the edge, narrow side across the fingers
                from mobile_env import square_box
                square_box(env.objects[name], STATIONS[st][1], x, y)
        for xy in [(3.6, 2.0), (3.6, -2.0), (-3.6, 2.0), (-3.6, -2.0)]:
            shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
            p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
        env.add_landing_pad()
        env.drone = Drone(env, start=(-0.6, 0.6, 1.5), wind=wind)   # wind is hidden from the AI
        env.record = dict(record)
        env.reports = []
        env.main_view = "station"
        if gui:
            p.resetDebugVisualizerCamera(5.0, 45, -50, [0.0, 0.0, 0.0])
        env.step(240)
        return env
    return build


def at_station(env, item, place):
    x, y, z = env.get_pos(item)
    cx, cy = env.station_centers[place]
    return abs(x - cx) < 0.27 and abs(y - cy) < 0.32 and z > TABLE_TOP


def audit_check(env):
    fixed = at_station(env, "soup can", "station C")
    kept = at_station(env, "banana", "station B")
    text = " ".join(env.reports).lower()
    reported = "drill" in text
    landed = env.drone.latched
    ok = fixed and kept and reported and landed
    miss = env.drone.touchdown_miss
    acc = f" (touchdown {miss * 100:.1f} cm from the pad center)" if miss is not None else ""
    return ok, (f"soup can back at station C: {fixed}; banana still at station B: {kept}; "
                f"missing power drill reported: {reported}; drone landed on the robot: {landed}{acc}")
