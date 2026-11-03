"""Every skill the team can do on a spoken order: the scene, the independent checker, the task
text for Muse Spark, and a scripted plan (no Spark) used for quick tests and as a fallback."""
import math
import random

import pybullet as p

from audit_tasks import AUDIT_DOC, AUDIT_METHODS, AUDIT_TEXT, AuditAPI, at_station, audit_check, audit_scene
from mission_tasks import mission_scene
from team_skills import TEAM_DOC, TEAM_METHODS, TeamAPI
from store_tasks import CATEGORIES, STORE_ITEMS, sort_check, store_scene
from throw_skills import THROW_DOC, THROW_METHODS, THROW_TEXT, ThrowAPI, throw_check, throw_scene

ITEMS = ["banana", "soup can", "mustard bottle", "foam brick", "gelatin box",
         "pear", "strawberry", "scissors", "medium clamp"]      # (the chips can is too wide for the gripper)
PLACES = ["station A", "station B", "station C"]
TEAM_CONTEXT = (" The drone can scout the stations; the ground robot carries one item at a time. "
                "Wind may be blowing.")
LAND_TEXT = ("The ground robot drives away, and the drone must land on the landing pad on top of the "
             "moving ground robot. Wind is blowing and gusts keep changing all the time.")


# ---------- scenes and checkers ----------
def land_scene(wind=(0.6, -0.4), gusts=1.0, seed=7):
    def build(gui=False):
        env = mission_scene(wind=wind)(gui=gui)
        env.drone.set_gusts(gusts, seed=seed)       # hidden, changing gusts
        return env
    return build


def landed_check(env):
    d = env.drone
    acc = f" (touchdown {d.touchdown_miss * 100:.1f} cm from the pad center)" if d.touchdown_miss is not None else ""
    return d.latched, f"drone landed on the moving robot: {d.latched}{acc}"


def move_check(item, place):
    def check(env):
        ok = at_station(env, item, place)
        x, y, z = env.get_pos(item)
        return ok, f"{item} on {place}'s table: {ok} (now at [{x:.2f}, {y:.2f}, {z:.2f}])"
    return check


def deliver_check(items):
    def check(env):
        from store_tasks import in_tray
        inside = [t for t in items if in_tray(env, t)]
        return len(inside) == len(items), f"in tray: {inside or 'nothing'} of {items}"
    return check


# ---------- real-world faults (hidden from the AI): slipping grasps, items moved by people ----------
def free_spot(env, place):
    """A free place on a station table (for the 'worker' who moves an item)."""
    cx, cy = env.station_centers[place]
    yaw = env.station_yaws[place]
    dx, dy = math.cos(yaw), math.sin(yaw)
    edge = 0.25 if abs(dx) > 0.5 else 0.30
    others = [env.get_pos(n) for n in env.objects if n not in env.static and n != "drone"]
    for depth in (0.07, 0.12, 0.17):
        for k in (0.0, -0.15, 0.15, -0.08, 0.08):
            d = edge - depth
            x, y = cx - dx * d - dy * k, cy - dy * d + dx * k
            if all(math.hypot(x - o[0], y - o[1]) > 0.13 for o in others):
                return x, y
    return None


def fault_scene(items, slip=0.3, move=True, seed=0):
    """The normal warehouse, plus faults: each grasp slips with probability `slip`, and (if `move`)
    a worker moves one of the ordered items to another station right after the drone has scanned."""
    def build(gui=False):
        env = mission_scene()(gui=gui)
        rng = random.Random(seed)
        env.slip_prob = slip
        env.fault_rng = random.Random(seed + 100)
        env.fault_log = []
        if move:
            item = rng.choice(list(items))
            x0, y0, z0 = env.get_pos(item)
            here = min(PLACES, key=lambda pl: math.dist(env.station_centers[pl], (x0, y0)))
            dest = rng.choice([pl for pl in PLACES if pl != here])
            go_to = env.go_to

            def go_to_with_worker(name):
                if not env.fault_log:                       # the first drive: the worker acts
                    spot = free_spot(env, dest)
                    if spot:
                        body = env.objects[item]
                        orn = p.getBasePositionAndOrientation(body)[1]
                        p.resetBasePositionAndOrientation(body, [spot[0], spot[1], z0 + 0.01], orn)
                        env.fault_log.append(f"a worker moved the {item} from {here} to {dest}")
                        env.say(f"a worker moved the {item}!")
                    else:
                        env.fault_log.append("no free spot for the worker")
                return go_to(name)
            env.go_to = go_to_with_worker
        return env
    return build


def _view(build, view):
    """Scene whose main camera is a close-up of one table (the overview camera only shows the packing table,
    so a goal picture of a change on station B or C must come from a camera that sees that table)."""
    def b(gui=False):
        env = build(gui)
        env.main_view = view
        return env
    return b


def start_layout(items):
    """Where every item starts, in words (from the scene's layout table)."""
    from mission_tasks import LAYOUT
    from store_tasks import STORE_LAYOUT
    lay = STORE_LAYOUT if all(i in STORE_ITEMS for i in items) else LAYOUT

    def side(k):                                  # slot 0 / 1 / 2 or metres sideways (+ = the robot's left)
        u = {0: -0.15, 1: 0.0, 2: 0.15}.get(k, k) if isinstance(k, int) else k
        return "left" if u > 0.06 else ("right" if u < -0.06 else "middle")
    return {it: f"on the table of {st} ({side(k)} side)" for it, (st, k) in lay.items()}


def custom_goal(order, start):
    """Muse Spark predicts where every item will be after a NEW job, so Muse Image draws exactly those items
    in exactly those places (not a generic 'fruit') and the close-up cameras look at the right tables."""
    import json
    import re
    places = [f"on the table of {st} ({sd} side)" for st in ("station A", "station B", "station C")
              for sd in ("left", "middle", "right")] + ["in the tray (on the packing station)", "on the top shelf"]
    try:
        from llm import spark
        q = (f'Warehouse order: "{order}"\nWhere the items are now: '
             + "; ".join(f"{k}: {v}" for k, v in start.items())
             + f"\nPossible places: {', '.join(places)}.\nWhere will EVERY item be after the job? Items the order "
               'does not mention stay where they are. Reply with JSON only: {"item": "place", ...}')
        text, _ = spark([{"role": "user", "content": q}], effort="low")
        after = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        after = {k: v for k, v in after.items() if k in start and v in places}
        if not after:
            raise ValueError("empty")
    except Exception:
        return f"The warehouse after this job is done: {order}. Do not add any new objects.", []
    full = {k: after.get(k, v) for k, v in start.items()}
    goal = ("The same warehouse after the job. Exactly these objects, each in its place: "
            + "; ".join(f"the {k} {v}" for k, v in full.items())
            + ". Do not add, remove or invent any other object.")
    changed = {v for k, v in full.items() if v != start[k]} | {start[k] for k, v in full.items() if v != start[k]}
    extra = []
    for st in ("station A", "station B", "station C"):
        if any(c.startswith(f"on the table of {st}") for c in changed):
            here = [f"{k} ({v.split('(')[-1].rstrip(')')})" for k, v in full.items()
                    if v.startswith(f"on the table of {st}")]
            extra.append((f"table:{st}", "Close-up photo of this dark grey table (keep the table exactly where it "
                          "is in the picture) with exactly these objects lying ON TOP of it: "
                          + (", ".join(here) or "nothing") + ". Nothing else on the table, nothing on the floor."))
    return goal, extra[:2]


def sort_goal(groups):
    """Name every item exactly, so the imagined picture does not invent extra things (an apple...)."""
    def one(it):
        return "a pair of scissors" if it == "scissors" else f"one {it}"

    def a(items):
        return ", ".join(one(it) for it in items[:-1]) + (" and " if len(items) > 1 else "") + one(items[-1])
    parts = [f"{a(g['items'])} inside the green tray on the packing table" if g["to"] == "tray"
             else f"{a(g['items'])} lying on the table of {g['to']}" for g in groups]
    return "; ".join(parts) + ". Only these objects, nothing else in the tray."


# ---------- one entry per skill: scene, checker, task text, goal picture, robot API ----------
TEAM = {"api_cls": TeamAPI, "api_doc": TEAM_DOC, "methods": TEAM_METHODS}


FAULT_TEXT = (" Real-world conditions are ON: a grasp can slip, and workers may move items after the "
              "drone has scanned them. Check every result and recover until the order is done.")


def scene_for(items):
    """The new store (fruit, tools, snacks) when it has every item, else the first warehouse."""
    return store_scene() if all(i in STORE_ITEMS for i in items) else mission_scene()


def task_for(skill, order, items=None, place=None, faults=False, seed=0, groups=None):
    said = f'Spoken order from the warehouse operator: "{order}". '
    if skill == "sort":
        groups = groups or [{"items": CATEGORIES["fruit"], "to": "tray"},
                            {"items": CATEGORIES["tools"], "to": "station A"}]
        jobs = "; ".join(f"the {' and the '.join(g['items'])} -> "
                         + ("into the tray at the packing station" if g["to"] == "tray" else f"onto {g['to']}'s table")
                         for g in groups)
        every = [i for g in groups for i in g["items"]]
        return dict(TEAM, task=said + f"Sort these items: {jobs}. Use place_on('tray') for the tray and "
                    "put_aside() to put an item down on a station's table." + TEAM_CONTEXT,
                    scene=scene_for(every), check=sort_check(groups),
                    goal=sort_goal([g for g in groups if g["to"] == "tray"] or groups),   # overview camera: the tray
                    extra_goals=[(f"table:{g['to']}", "Close-up photo of this table with "
                                  + " and ".join(g["items"]) + " lying on top of it. Everything else stays the same.")
                                 for g in groups if g["to"] != "tray"])
    if skill == "custom":                       # a NEW job: no hand-written skill, no hand-written checker
        from store_tasks import judge_check, with_start
        scene = scene_for(items or [])
        start = start_layout(items or [])
        goal, extra = custom_goal(order, start)
        return dict(TEAM, task=said + "This is a NEW job that the team has no ready-made skill for. Plan it "
                    "yourself from the robot actions: find the items (drone_scan, recall; if the drone misses an "
                    "item, visit the tables and use sees), carry them one at a time (go_to, pick, then "
                    "place_on('tray') or put_aside() on a station's table), and check every result. If the order "
                    "also asks to land the drone on the moving robot, call land_on_moving_robot() at the end. "
                    "Do exactly what the operator asked; leave other items where they are. "
                    f"Where the items start: {'; '.join(f'{k}: {v}' for k, v in start.items())}. "
                    "Afterwards an independent judge compares where every item was before and after."
                    + TEAM_CONTEXT,
                    scene=with_start(scene if not extra else _view(scene, extra[0][0])), check=judge_check(order),
                    goal=(extra[0][1] if extra else goal), extra_goals=extra[1:])
    if skill == "relay":
        item = (items or ["strawberry"])[0]
        place = place if place in PLACES else "station C"
        return dict(TEAM, task=said + f"The {item} is on the top shelf (1.3 m high), out of the arm's reach. "
                    f"Put it on the table of {place}. Team plan: the drone grabs it, then hands it to the ground "
                    "robot ON THE MOVE (handoff_to_robot drops it into the basket on the robot's back while the "
                    "robot drives to the place), then the arm takes it out of the basket and puts it on the table."
                    + TEAM_CONTEXT,
                    scene=store_scene(shelf_items=(item,), basket=True), check=relay_check(item, place),
                    goal=f"One {item} lying on the table of {place} in the warehouse, a wheeled robot with an "
                         "orange basket on its back parked in front of the table, a small drone flying above.",
                    extra_goals=[(f"table:{place}", f"Close-up photo of this table with one {item} lying on top "
                                                    "of it. Everything else stays the same.")])
    if skill == "air":
        items = (items or ["strawberry"])[:2]
        return dict(TEAM, task=said + f"The {' and the '.join(items)} are on the top shelf (1.3 m high), out of "
                    "the arm's reach. Bring them into the tray at the packing station." + TEAM_CONTEXT,
                    scene=store_scene(shelf_items=tuple(items)), check=deliver_check(items),
                    goal=f"The {' and the '.join(items)} are inside the green tray on the packing station table.")
    if skill == "deliver":
        items = items or ["banana"]
        return dict(TEAM, task=said + f"Bring the {' and the '.join(items)} to the packing station and put "
                    "them in the tray." + TEAM_CONTEXT + " The ground robot does not need to drive away afterwards."
                    + (FAULT_TEXT if faults else ""),
                    scene=fault_scene(items, seed=seed) if faults else scene_for(items), check=deliver_check(items),
                    goal=f"The {' and the '.join(items)} are inside the green tray on the packing station table.")
    if skill == "move":
        item, place = (items or ["banana"])[0], place or "station C"
        return dict(TEAM, task=said + f"Move the {item} to {place}: find it, pick it up, drive to {place} and "
                    "put it down on that table with put_aside()." + TEAM_CONTEXT,
                    scene=scene_for([item]), check=move_check(item, place),
                    goal=f"The {item} lies on the table of {place} in the warehouse.",
                    extra_goals=[(f"table:{place}", f"Close-up photo of this table with the {item} lying on top "
                                                    "of it. Everything else stays the same.")])
    if skill == "throw":
        return dict(api_cls=ThrowAPI, api_doc=THROW_DOC, methods=THROW_METHODS, task=said + THROW_TEXT,
                    scene=throw_scene(), check=throw_check,
                    goal="A yellow tennis ball inside a blue bin on a warehouse floor.")
    if skill == "audit":
        return dict(api_cls=AuditAPI, api_doc=AUDIT_DOC, methods=AUDIT_METHODS, task=said + AUDIT_TEXT,
                    scene=audit_scene(), check=audit_check, goal=None)
    if skill == "land":
        return dict(TEAM, task=said + LAND_TEXT, scene=land_scene(), check=landed_check,
                    goal="A small drone sitting on an orange pad on top of a wheeled robot.")
    raise ValueError(f"unknown skill {skill}")


# ---------- scripted plans (no Spark): what a careful human programmer would write ----------
def plan_deliver(api, items):
    for st in [pl for pl in api.places() if pl != "packing station"]:
        api.drone_scan(st, items)
    for it in items:
        where = api.recall(it)
        if where is None:
            continue
        api.go_to(where)
        if api.pick(it):
            api.go_to("packing station")
            api.place_on("tray")


def plan_deliver_verified(api, items):
    """Same order, but every step is checked: retry a slipped grasp, re-scan when an item is gone."""
    stations = [pl for pl in api.places() if pl != "packing station"]
    for st in stations:
        api.drone_scan(st, items)
    for it in items:
        for _ in range(5):
            where = api.recall(it)
            if where is None:                      # not on the map (or the map was stale): look again
                for st in stations:
                    if it in api.drone_scan(st, [it]):
                        where = st
                        break
            if where is None:
                break
            api.go_to(where)
            if api.pick(it):
                break
        if api.is_holding():
            api.go_to("packing station")
            api.place_on("tray")


def fetch(api, it, stations):
    """Find the item (drone map first, re-scan if needed), drive there and pick it, with retries."""
    for _ in range(5):
        where = api.recall(it)
        if where is None:
            for st in stations:
                if it in api.drone_scan(st, [it]):
                    where = st
                    break
        if where is None:
            return False
        api.go_to(where)
        if api.pick(it):
            return True
    return False


def plan_sort(api, groups):
    stations = [pl for pl in api.places() if pl != "packing station"]
    every = [i for g in groups for i in g["items"]]
    for st in stations:
        api.drone_scan(st, every)
    for g in groups:
        for it in g["items"]:
            if fetch(api, it, stations):
                if g["to"] == "tray":
                    api.go_to("packing station")
                    api.place_on("tray")
                else:
                    api.go_to(g["to"])
                    api.put_aside()


def plan_relay(api, item, place):
    """Drone grabs from the top shelf -> hand-off into the moving robot's basket -> arm puts it on the table."""
    for _ in range(2):
        api.drone_fly_over("top shelf")
        if api.drone_grab(item):
            break
    if api.handoff_to_robot(place):
        for _ in range(2):
            if api.take_from_basket(item):
                api.put_aside()
                return


def relay_check(item, place):
    from store_tasks import on_station

    def check(env):
        ok = on_station(env, item, place)
        x, y, z = env.get_pos(item)
        return ok, f"{item} on {place}'s table: {ok} (now at [{x:.2f}, {y:.2f}, {z:.2f}])"
    return check


def plan_air(api, items):
    for it in items:
        for _ in range(2):
            api.drone_fly_over("top shelf")
            if api.drone_grab(it):
                api.drone_fly_over("packing station")
                api.drone_drop_into("tray")
                break


def plan_move(api, item, place):
    where = api.recall(item)
    if where is None:
        for st in [pl for pl in api.places() if pl != "packing station"]:
            if item in api.drone_scan(st, [item]):
                where = st
                break
    if where is None:
        return
    api.go_to(where)
    if api.pick(item):
        api.go_to(place)
        api.put_aside()


def plan_throw(api):
    api.go_to("ball rack")
    api.pick("tennis ball")
    api.go_to("throw line")
    b, rp = api.see("blue bin"), api.release_point()
    target = math.hypot(b[0] - rp[0], b[1] - rp[1])
    scale = 1.0                       # real distance / predicted distance (learned from each throw)
    for _ in range(3):
        lo, hi = 1.0, 6.0
        for _ in range(40):           # find the speed whose (corrected) distance hits the target
            mid = (lo + hi) / 2
            if api.predict_distance(mid, 45, 0.15) * scale < target:
                lo = mid
            else:
                hi = mid
        r = api.throw(mid, 45)
        if r["in_bin"]:
            return
        scale = r["distance"] / api.predict_distance(mid, 45, 0.15)
        api.go_to("ball rack")
        if not api.pick("tennis ball"):
            return
        api.go_to("throw line")


def plan_audit(api):
    record = api.inventory_record()
    for st in [pl for pl in api.places() if pl != "packing station"]:
        api.drone_scan(st, list(record))
    seen = api.known_map()
    for item, should in record.items():
        where = seen.get(item)
        if where is None:
            api.report(f"missing: {item} (records say {should})")
        elif where != should:
            api.go_to(where)
            if api.pick(item):
                api.go_to(should)
                api.put_aside()
    plan_land(api)


def plan_land(api, rl=True):
    api.start_driving()
    g = api.ground_robot_position()
    pos = api.drone_fly_to(g[0], g[1], 1.3)
    api.drone_set_wind_correction(-6 * (pos[0] - g[0]), -6 * (pos[1] - g[1]))
    p1 = api.drone_find_pad()
    g = api.ground_robot_position()
    api.drone_fly_to(g[0] - 0.38 + 0.6, g[1], 1.3)
    p2 = api.drone_find_pad()
    vx, vy = ((p2[0] - p1[0]) / 3.0, (p2[1] - p1[1]) / 3.0) if p1 and p2 else (0.2, 0.0)
    return api.drone_land_rl(vx, vy) if rl else api.drone_land_tracking(vx, vy)


def run_plan(skill, api, items=None, place=None, verified=True, groups=None):
    if skill == "relay":
        plan_relay(api, (items or ["strawberry"])[0], place if place in PLACES else "station C")
    elif skill == "air":
        plan_air(api, items or ["strawberry"])
    elif skill == "sort":
        plan_sort(api, groups or [{"items": CATEGORIES["fruit"], "to": "tray"},
                                  {"items": CATEGORIES["tools"], "to": "station A"}])
    elif skill == "deliver":
        (plan_deliver_verified if verified else plan_deliver)(api, items or ["banana"])
    elif skill == "move":
        plan_move(api, (items or ["banana"])[0], place or "station C")
    elif skill == "throw":
        plan_throw(api)
    elif skill == "audit":
        plan_audit(api)
    elif skill == "land":
        plan_land(api)
