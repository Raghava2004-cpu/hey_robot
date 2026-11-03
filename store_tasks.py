"""A bigger warehouse with new everyday objects (fruit, tools, snacks) for reasoning orders
("bring me some fruit") and sorting jobs ("fruit in the tray, tools on station A")."""
import math
import os

import pybullet as p

from drone import Drone
from mission_tasks import STATIONS, YCB, YCB_FILES
from mobile_env import ALIASES, MobileEnv

NEW_FILES = {"pear": "YcbPear", "strawberry": "YcbStrawberry", "scissors": "YcbScissors",
             "medium clamp": "YcbMediumClamp", "chips can": "YcbChipsCan"}
YCB_FILES.update(NEW_FILES)                       # the drone's product list now knows them too
ALIASES.update({"pear": ["pear", "green pear", "yellow pear"],
                "strawberry": ["strawberry", "red strawberry"],
                "scissors": ["scissors", "pair of scissors"],
                "medium clamp": ["clamp", "medium clamp", "black clamp"],
                "chips can": ["chips can", "pringles can", "tall red can"]})

# what is on each table: (station, slot 0/1/2 along the front edge)
# a whole number = one of the 3 slots; a decimal = metres sideways from the table's middle (more room
# for long items: a banana pushed down by the gripper must not tip the scissors next to it off the table)
STORE_LAYOUT = {"gelatin box": ("station A", 0), "strawberry": ("station A", 2),
                "banana": ("station B", 0), "scissors": ("station B", 0.12),
                "pear": ("station C", 0), "medium clamp": ("station C", 1), "soup can": ("station C", 2)}
STORE_ITEMS = list(STORE_LAYOUT)
CATEGORIES = {"fruit": ["banana", "pear", "strawberry"],
              "food": ["banana", "pear", "strawberry", "gelatin box", "soup can"],
              "tools": ["scissors", "medium clamp"]}


SHELF_CENTER, SHELF_TOP = (1.5, -1.5), 1.3      # a high shelf: out of the arm's reach


def _spot(info, station, k):
    """Slot k (0, 1, 2) on the front edge, or k metres sideways from the middle of the front edge."""
    if isinstance(k, int):
        return info["items"][k]
    (x0, y0), (x2, y2) = info["items"][0], info["items"][2]     # slots 0 and 2 sit at -0.15 and +0.15
    mx, my = (x0 + x2) / 2, (y0 + y2) / 2
    ux, uy = (x2 - x0) / 0.30, (y2 - y0) / 0.30
    return mx + ux * k, my + uy * k


def _along_edge(body, x, y, yaw):
    """Turn a long item (scissors) so it lies along the table's front edge, not sticking out over it
    (an item that sticks out gets bumped off the table by the robot's front)."""
    lo, hi = p.getAABB(body)
    across = abs(math.cos(yaw)) > 0.5                    # robot faces along x: the edge runs along y
    if (hi[0] - lo[0] > hi[1] - lo[1]) == across:        # the long side points at the robot: turn 90 degrees
        pos, orn = p.getBasePositionAndOrientation(body)
        orn = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2]), [0, 0, 0], orn)[1]
        p.resetBasePositionAndOrientation(body, pos, orn)
    lo, hi = p.getAABB(body)                             # put the item's middle (not its base point) on the slot
    pos, orn = p.getBasePositionAndOrientation(body)
    p.resetBasePositionAndOrientation(body, [pos[0] + x - (lo[0] + hi[0]) / 2,
                                             pos[1] + y - (lo[1] + hi[1]) / 2, pos[2]], orn)


def store_scene(wind=(0.8, 0.5), layout=None, shelf_items=(), basket=False):
    """shelf_items = light items that start on the high shelf (only the drone can reach them)."""
    layout = {k: v for k, v in (layout or STORE_LAYOUT).items() if k not in shelf_items}

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
            x, y = _spot(info[st], st, k)
            body = p.loadURDF(os.path.join(YCB, YCB_FILES[name], "model.urdf"), [x, y, info[st]["top"] + 0.06])
            p.changeDynamics(body, -1, rollingFriction=0.01, spinningFriction=0.01)   # round fruit stays put
            if name == "gelatin box":                     # square to the edge, narrow side across the fingers
                from mobile_env import square_box
                square_box(body, env.station_yaws[st], x, y)
            else:
                _along_edge(body, x, y, env.station_yaws[st])
            env.objects[name] = body
        # the high shelf
        half = [0.2, 0.2, SHELF_TOP / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=(0.55, 0.6, 0.7, 1))
        env.objects["top shelf"] = p.createMultiBody(0, col, vis, [SHELF_CENTER[0], SHELF_CENTER[1], SHELF_TOP / 2])
        env.static.add("top shelf")
        env.station_centers["top shelf"] = SHELF_CENTER
        env.station_yaws["top shelf"] = -math.pi / 4
        env.station_heights = {"top shelf": SHELF_TOP}
        for k, name in enumerate(shelf_items):
            body = p.loadURDF(os.path.join(YCB, YCB_FILES[name], "model.urdf"),
                              [SHELF_CENTER[0] - 0.08 + 0.16 * k, SHELF_CENTER[1], SHELF_TOP + 0.05])
            p.changeDynamics(body, -1, rollingFriction=0.01, spinningFriction=0.01)
            env.objects[name] = body
        for xy in [(3.6, 2.0), (3.6, -2.0), (-3.6, 2.0), (-3.6, -2.0)]:
            shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
            p.resetBasePositionAndOrientation(shelf, [xy[0], xy[1], 0], [0, 0, 0, 1])
        env.add_landing_pad(basket=basket)
        env.drone = Drone(env, start=(-0.6, 0.6, 1.5), wind=wind)
        env.main_view = "station"
        env.label_places()
        if gui:
            p.addUserDebugText("TOP SHELF", [SHELF_CENTER[0], SHELF_CENTER[1], SHELF_TOP + 0.5],
                               textColorRGB=[0.05, 0.2, 0.55], textSize=1.4)
        if gui:
            p.resetDebugVisualizerCamera(5.0, 45, -50, [0.0, 0.0, 0.0])
        env.step(240)
        env.start_pos = {n: env.get_pos(n) for n in env.objects if n not in env.static}
        return env
    return build


def in_tray(env, item):
    """The item's middle (its bounding box, not its base point: a banana's base point is off-center)
    is inside the tray, and the item rests on the tray, not on the floor."""
    tray = env.get_pos("tray")
    lo, hi = p.getAABB(env.objects[item])
    x, y = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    return abs(x - tray[0]) < 0.15 and abs(y - tray[1]) < 0.15 and lo[2] > 0.30


def on_station(env, item, place):
    """The item rests on that station's table (judged by its bounding box, not its base point)."""
    lo, hi = p.getAABB(env.objects[item])
    x, y = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    cx, cy = env.station_centers[place]
    return abs(x - cx) < 0.28 and abs(y - cy) < 0.33 and lo[2] > 0.40


def place_of(env, name):
    """Where an item is, in words (from the simulator: used by the independent judge of new jobs)."""
    if env.held and env.held[0] == name:
        return "held by the robot arm"
    d = getattr(env, "drone", None)
    if d is not None and d.cargo and d.cargo[0] == name:
        return "carried by the drone"
    lo, hi = p.getAABB(env.objects[name])
    x, y, z = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]
    if "tray" in env.objects and in_tray(env, name):
        return "in the tray (on the packing station)"
    if getattr(env, "basket", False):
        bx, by, _ = env.basket_pos()
        if abs(x - bx) < 0.22 and abs(y - by) < 0.22 and z > 0.35:
            return "in the basket on the robot's back"
    if "top shelf" in getattr(env, "station_centers", {}):
        sx, sy = env.station_centers["top shelf"]
        if abs(x - sx) < 0.22 and abs(y - sy) < 0.22 and z > SHELF_TOP - 0.05:
            return "on the top shelf"
    for st, (cx, cy) in getattr(env, "station_centers", {}).items():
        if st != "top shelf" and abs(x - cx) < 0.28 and abs(y - cy) < 0.33 and z > 0.40:
            from team_skills import table_side
            side = table_side(env, st, x, y)
            return f"on the table of {st}" + (f" ({side} side)" if side else "")
    return "on the floor" if z < 0.30 else "somewhere else"


def items_of(env):
    return [n for n in env.objects if n not in env.static and n != "drone" and "tennis ball" not in n]


def drone_state(env):
    d = getattr(env, "drone", None)
    if d is None:
        return "no drone"
    if d.latched:
        return "landed on the moving robot"
    return "flying (not landed)"


def with_start(build):
    """Scene builder that also remembers where every item started (for the judge of new jobs)."""
    def b(gui=False):
        env = build(gui)
        env.start_places = {n: place_of(env, n) for n in items_of(env)}
        env.start_places["the drone"] = drone_state(env)
        return env
    return b


def judge_check(order):
    """Checker for a NEW job that has no hand-written checker: an independent Muse Spark call sees only the
    order and where every item was before and after (from the simulator), never the robots' program."""
    def check(env):
        start = getattr(env, "start_places", {})
        final = {n: (drone_state(env) if n == "the drone" else place_of(env, n)) for n in start}
        before = "; ".join(f"{n}: {w}" for n, w in start.items())
        after = "; ".join(f"{n}: {w}" for n, w in final.items())
        moved = [n for n in start if start[n] != final[n]]
        q = (f'Order given to a warehouse robot team: "{order}"\n'
             f"BEFORE the job: {before}\nAFTER the job: {after}\n"
             "You are an independent judge. Using only these positions, did the robots fully carry out the "
             "order (every part of it)? Items the order does not mention should stay on the same table. The side "
             "of a table (left/middle/right, seen by the robot) only matters when the order is about places on the "
             "SAME table; otherwise being on the right table is enough. If the order was already true at the start, "
             "say DONE and mention it. Reply DONE or NOT DONE on the first line, then one short reason.")
        try:
            from llm import spark
            text, _ = spark([{"role": "user", "content": q}], effort="low")
            first = text.strip().splitlines()[0].upper() if text.strip() else ""
            ok = first.startswith("DONE") or (first.startswith("**DONE") and "NOT" not in first)
            reason = " ".join(text.strip().splitlines()[1:])[:160] or first
        except Exception as e:
            ok, reason = False, f"judge unavailable ({type(e).__name__})"
        return ok, (f"Muse Spark judge: {'DONE' if ok else 'NOT DONE'} - {reason} | moved: {moved or 'nothing'} | "
                    f"after: {after}")
    return check


def sort_check(groups):
    """groups = [{"items": [...], "to": "tray" or "station A/B/C"}, ...]"""
    at_station = on_station

    def check(env):
        good, bad = [], []
        for g in groups:
            for it in g["items"]:
                ok = in_tray(env, it) if g["to"] == "tray" else at_station(env, it, g["to"])
                (good if ok else bad).append(f"{it}->{g['to']}")
        named = [it for g in groups for it in g["items"]]
        moved = [n for n, q in getattr(env, "start_pos", {}).items()
                 if n not in named and n != "drone" and n in env.objects
                 and math.dist(env.get_pos(n)[:2], q[:2]) > 0.05]
        msg = f"sorted correctly: {good or 'none'}; not yet: {bad or 'none'}"
        if moved:
            msg += f"; knocked over by mistake (should not move): {moved}"
        return not bad and not moved, msg
    return check
