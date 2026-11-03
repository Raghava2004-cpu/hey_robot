import math

import pybullet as p

from mobile_env import MobileEnv


def mobile_scene(hidden=False):
    def build(gui=False):
        env = MobileEnv(gui=gui)
        top = env.add_table("pickup table", (1.6, 0.0))
        if hidden:
            env.add_cube("red", (1.42, 0.08), top, [1, 0, 0, 1])
            env.step(60)
            env.add_cube_on("blue", "red", [0, 0, 1, 1])           # blue hides red
            env.add_cube("yellow", (1.45, -0.15), top, [1, 0.9, 0, 1])
        else:
            env.add_cube("red", (1.42, 0.08), top, [1, 0, 0, 1])
            env.add_cube("blue", (1.42, -0.10), top, [0, 0, 1, 1])
        pack = env.add_table("packing station", (0.0, -1.6), rgba=(0.6, 0.5, 0.4, 1))
        env.add_tray("tray", (0.0, -1.42), pack)
        for xy in [(3.0, 1.5), (3.0, -1.5)]:
            shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
            p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
        env.add_location("pickup table", (0.75, 0.0, 0.0),
                         spots=[(1.45, 0.22, top), (1.45, -0.22, top), (1.55, 0.0, top)])
        env.add_location("packing station", (0.0, -0.70, -math.pi / 2),
                         spots=[(0.18, -1.42, pack), (-0.18, -1.42, pack)])
        
        env.main_view = "station"   # imagination check watches the packing station
        env.step(120)
        return env
    return build


def red_in_tray(env):
    tray, red = env.get_pos("tray"), env.get_pos("red")
    ok = abs(red[0] - tray[0]) < 0.1 and abs(red[1] - tray[1]) < 0.1
    return ok, "red cube is in the tray" if ok else "red cube is NOT in the tray"


TEXT = "Bring the red cube to the packing station and put it in the tray."
MOBILE_TASKS = {
    "fetch": (TEXT, mobile_scene(hidden=False), red_in_tray),
    "fetch_hidden": (TEXT, mobile_scene(hidden=True), red_in_tray),
}