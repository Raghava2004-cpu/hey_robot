import math
import os

import pybullet as p

from drone import Drone
from mobile_env import ALIASES, MobileEnv

YCB = os.path.join("third_party", "pybullet-object-models", "pybullet_object_models", "ycb_objects")
YCB_FILES = {"banana": "YcbBanana", "soup can": "YcbTomatoSoupCan", "foam brick": "YcbFoamBrick",
             "gelatin box": "YcbGelatinBox", "mustard bottle": "YcbMustardBottle"}
ALIASES.update({"soup can": ["soup can", "tomato soup can", "can"],
                "foam brick": ["foam brick", "red brick"],
                "gelatin box": ["gelatin box", "small box"],
                "mustard bottle": ["mustard bottle", "yellow bottle"]})

STATIONS = {"station A": ((1.6, 0.0), 0.0),
            "station B": ((0.0, 1.6), math.pi / 2),
            "station C": ((-1.6, 0.0), math.pi)}
LAYOUT = {"foam brick": ("station A", 0), "gelatin box": ("station A", 2),
          "banana": ("station B", 1),
          "soup can": ("station C", 0), "mustard bottle": ("station C", 2)}


def mission_scene(wind=(0.8, 0.5), layout=LAYOUT):
    def build(gui=False):
        env = MobileEnv(gui=gui)
        info = {n: env.add_station(n, c, yaw) for n, (c, yaw) in STATIONS.items()}
        env.station_centers = {n: c for n, (c, yaw) in STATIONS.items()}
        env.station_yaws = {n: yaw for n, (c, yaw) in STATIONS.items()}
        pk = env.add_station("packing station", (0.0, -1.6), -math.pi / 2, rgba=(0.45, 0.35, 0.25, 1))
        env.station_centers["packing station"] = (0.0, -1.6)
        env.station_yaws["packing station"] = -math.pi / 2
        env.add_tray("tray", (0.0, -1.45), pk["top"], size=0.3)
        for name, (st, k) in layout.items():
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
        env.main_view = "station"
        if gui:
            p.resetDebugVisualizerCamera(5.0, 45, -50, [0.0, 0.0, 0.0])
        env.step(240)
        return env
    return build


def mission_check(targets):
    def check(env):
        tray = env.get_pos("tray")
        inside = [t for t in targets
                  if abs(env.get_pos(t)[0] - tray[0]) < 0.15 and abs(env.get_pos(t)[1] - tray[1]) < 0.15]
        landed = env.drone.latched
        ok = len(inside) == len(targets) and landed
        miss = env.drone.touchdown_miss
        acc = f" (touchdown {miss * 100:.1f} cm from the pad center)" if miss is not None else ""
        return ok, f"in tray: {inside or 'nothing'} of {targets}; drone landed on the robot: {landed}{acc}"
    return check


MISSION_TEXT = ("Mission: use the drone to find the banana and the soup can in the warehouse. "
                "The ground robot must bring both to the packing station and put them in the tray. "
                "Then the ground robot drives away, and the drone must land on the landing pad on top "
                "of the moving ground robot. Wind may be blowing.")
MISSIONS = {"mission1": (MISSION_TEXT, mission_scene(), mission_check(["banana", "soup can"]))}
