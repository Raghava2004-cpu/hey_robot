import math

import pybullet as p

from mobile_env import MobileEnv

COLORS = {"red": [1, 0, 0, 1], "blue": [0, 0, 1, 1], "yellow": [1, 0.9, 0, 1],
          "purple": [0.5, 0, 0.8, 1], "orange": [1, 0.5, 0, 1]}
STATIONS = {"station A": ((1.6, 0.0), 0.0),
            "station B": ((0.0, 1.6), math.pi / 2),
            "station C": ((-1.6, 0.0), math.pi)}
RESULT = {"meters": None}


def warehouse(layout, hidden=None):
    """layout: {color: (station, spot 0-2)}. hidden: (target, cover) puts cover on top of target."""
    def build(gui=False):
        env = MobileEnv(gui=gui)
        info = {n: env.add_station(n, c, yaw) for n, (c, yaw) in STATIONS.items()}
        pk = env.add_station("packing station", (0.0, -1.6), -math.pi / 2, rgba=(0.6, 0.5, 0.4, 1))
        env.add_tray("tray", (0.0, -1.42), pk["top"])
        for color, (st, k) in layout.items():
            env.add_cube(color, info[st]["items"][k], info[st]["top"], COLORS[color])
        if hidden:
            env.step(60)
            target, cover = hidden
            env.add_cube_on(cover, target, COLORS[cover])
        for xy in [(3.2, 1.6), (3.2, -1.6), (-3.2, 1.6), (-3.2, -1.6)]:
            shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
            p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
        env.main_view = "station"
        env.step(120)
        return env
    return build


def order_check(targets):
    def check(env):
        tray = env.get_pos("tray")
        inside = [t for t in targets
                  if abs(env.get_pos(t)[0] - tray[0]) < 0.1 and abs(env.get_pos(t)[1] - tray[1]) < 0.1]
        RESULT["meters"] = round(env.odometer, 1)
        ok = len(inside) == len(targets)
        return ok, f"in tray: {inside or 'nothing'} of {targets}; robot drove {env.odometer:.1f} m"
    return check


BASE = {"red": ("station C", 0), "blue": ("station A", 2),
        "yellow": ("station B", 1), "purple": ("station C", 1)}
MOVED = dict(BASE)
MOVED["blue"] = ("station B", 2)          # a worker moved the blue cube

ORDERS = {
    "order1": ("Order: bring the yellow cube and the purple cube to the packing station and put "
               "them in the tray. You do not know where they are.",
               warehouse(BASE, hidden=("purple", "orange")), order_check(["yellow", "purple"])),
    "order2": ("Order: bring the red cube to the packing station and put it in the tray.",
               warehouse(BASE), order_check(["red"])),
    "order3": ("Order: bring the blue cube to the packing station and put it in the tray.",
               warehouse(MOVED), order_check(["blue"])),
}