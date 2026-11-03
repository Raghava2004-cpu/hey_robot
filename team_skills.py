import json
import math
import os

TEAM_METHODS = {"places", "go_to", "pick", "place_on", "put_aside", "is_holding", "sees",
                "start_driving", "ground_robot_position", "recall", "known_map",
                "drone_scan", "drone_position", "drone_fly_to", "drone_find_pad",
                "drone_wait", "drone_set_wind_correction", "drone_land_at", "drone_land_tracking",
                "drone_land_rl", "drone_fly_over", "drone_grab", "drone_drop_into",
                "handoff_to_robot", "take_from_basket", "land_on_moving_robot"}
MAX_ACTIONS = 60
MAP_FILE = "mission_map.json"

TEAM_DOC = """
GROUND ROBOT (wheeled robot with an arm; carries ONE item at a time):
robot.places() -> list of place names, e.g. ["station A", "station B", "station C", "packing station"]
robot.go_to(place) -> True/False   (drive to a place and face its table)
robot.pick(item) -> True/False   (pick the named item from the table in front, e.g. "banana")
robot.place_on(thing) -> True/False   (put the held item on / into the named thing in front, e.g. "tray")
robot.put_aside(side=None) -> True/False   (gently put the held item down on a free spot of the table in front; side="left", "middle" or "right" (as seen by the robot) chooses where on that table, e.g. to swap two items that are on the SAME table. A table has room for about 3 items: to swap two items on a full table, first park one of them on ANOTHER table, then come back for it)
robot.is_holding() -> True/False
robot.sees(item) -> True/False   (the ground robot's front camera checks with SAM 3.1 whether the item is on the table in front)
robot.start_driving() -> True   (the ground robot drives to the packing station, turns to face +x and keeps driving forward at a constant slow speed; it only stops when the drone lands on it, so do all ground work first)
robot.ground_robot_position() -> [x, y] of the ground robot's center (rough odometry)
SHARED MAP:
robot.recall(item) -> the place where the drone or robot last saw the item, or None
robot.known_map() -> dict {item: place}
DRONE (flies above everything; it keeps hovering where it is while the ground robot works):
robot.drone_scan(place, items) -> list of the given items seen on that place's table. The drone flies near the table in 3.0 s and looks; for every candidate SAM 3.1 finds, it flies closer for a second look (3.0 s more each) and Muse Spark identifies the product. Updates the MAP.
robot.drone_position() -> [x, y, z]
robot.drone_fly_to(x, y, z) -> [x, y, z] where the drone is after EXACTLY 3.0 s of flying toward the target (z 0.5-2.0)
robot.drone_find_pad() -> [x, y] of the orange landing pad on the ground robot seen by the drone camera, or None (from 1.3 m it sees about 0.5 m around the drone). Takes no time.
robot.drone_wait(seconds) -> True   (max 5 seconds)
robot.drone_set_wind_correction(ax, ay) -> True   (extra acceleration in m/s^2 to cancel wind; limit +-3)
robot.drone_land_at(x, y) -> True if landed. OPEN-LOOP: the drone reaches PAD HEIGHT EXACTLY 2.0 s after starting, aiming at the fixed point (x, y). It lands only if the pad is under it at that moment (the pad is 0.45 m wide).
robot.drone_land_tracking(vx, vy) -> True if landed. CLOSED-LOOP: the drone catches up with the pad and follows it at its current height (up to 3 s), re-finding it with SAM 3.1 every 0.25 s and fitting its velocity, then descends (reaching pad height exactly 2.0 s later) while it keeps correcting its aim. (vx, vy) = your best estimate of the pad velocity in m/s (a starting guess). Call it with the pad in view, ideally right below the drone, from about 1.3 m height, after setting the wind correction.
robot.drone_land_rl(vx, vy) -> True if landed. LEARNED REFLEX: a neural-network policy trained with reinforcement learning (PPO, 4,096 simulated drones in random wind and gusts) flies the last 3 s: it follows the pad for 1.0 s, then descends for 2.0 s, re-finding the pad with SAM 3.1 every 0.25 s and feeling gusts from its own motion. Best when the wind keeps changing (gusts). Call it exactly like drone_land_tracking.
DRONE GRIPPER (a small vacuum gripper under the drone; light items only, up to 100 g):
robot.drone_fly_over(place) -> [x, y, z]   (fly 3.0 s to 2.0 m above a place, e.g. "top shelf" or "packing station")
robot.drone_grab(item) -> True/False   (the drone must be above the item: it aligns itself with SAM 3.1, comes down, grabs it, climbs back; refuses items heavier than 100 g)
robot.drone_drop_into(thing) -> True/False   (the drone must be above the thing, e.g. "tray": it aligns, comes down close and lets go)
Items on the "top shelf" (1.3 m high) are out of the arm's reach: only the drone can get them.
TEAM HAND-OFF (only when the ground robot has a basket on its back):
robot.handoff_to_robot(place) -> True/False   (the drone must be carrying an item. The ground robot drives toward the place; on the way the drone follows the robot's orange basket with SAM 3.1, comes down and drops the item into it WHILE THE ROBOT IS MOVING; the basket camera then checks with SAM 3.1 that the item landed; the robot parks at the place. If the item is not in the basket it returns False.)
robot.land_on_moving_robot() -> True/False   (COMPOSITE SKILL, use it when an order says to land the drone on the (moving) robot, e.g. at the END of a job: the robot drives off, the drone measures the pad's speed with SAM 3.1 and lands with the learned RL reflex)
robot.take_from_basket(item) -> True/False   (the arm turns around and picks the item out of the robot's own basket, using the basket camera and SAM 3.1. Then use put_aside() to put it on the table in front.)
DRONE AUTOPILOT LAW (exact): acceleration = 6*(target - position) + 4*(target_velocity - velocity) + gravity + wind_correction + UNKNOWN_WIND. In a steady hover: position - target = (wind_correction + UNKNOWN_WIND) / 6.
The landing pad is about 0.38 m BEHIND the ground robot's center. Time does NOT pass during drone_find_pad, recall, known_map, ground_robot_position, drone_position; it passes during all other actions.
REAL-WORLD NOTE: a grasp can slip (pick returns False and the item stays on the table), and people can move items after the drone scanned them (then the map is stale and pick cannot see the item). Check every result and recover: retry a slipped grasp, and re-scan with the drone when an item is gone. Never loop forever: if an item is still not found after two full scans, skip it and finish the other items.
Item names are short noun phrases like "banana", "soup can", "tray".
Math helpers: math.sqrt, math.sin, math.cos, math.atan2, math.hypot, math.pi
"""


def table_side(env, place, x, y):
    """'left' / 'middle' / 'right' of a station's table, as seen by the robot standing in front of it."""
    if place not in getattr(env, "station_centers", {}) or place not in getattr(env, "station_yaws", {}):
        return None
    cx, cy = env.station_centers[place]
    yaw = env.station_yaws[place]
    u = (x - cx) * -math.sin(yaw) + (y - cy) * math.cos(yaw)      # + = the robot's left
    return "left" if u > 0.06 else ("right" if u < -0.06 else "middle")


def sam_many(rgb, items, workers=6, max_names=3):
    """SAM 3.1 for several items at once (one request per item, sent in parallel; each item still tries
    its other names one after another). Returns {item: (best box, phrase) or None}."""
    from concurrent.futures import ThreadPoolExecutor
    from mobile_env import ALIASES
    from perception import sam_find

    def one(it):
        for ph in ALIASES.get(it, [it])[:max_names]:
            boxes = sam_find(rgb, ph)
            if boxes:
                return max(boxes, key=lambda b: b["conf"] or 0), ph
        return None
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return dict(zip(items, pool.map(one, items)))


# Where the drone waits between jobs: high up in a corner with no table and no driving lane under it,
# so the robot's arm can never hit it (the arm reaches about 1.0 m high).
DRONE_PARK = (-1.2, -1.2, 1.8)


class TeamAPI:
    def __init__(self, env):
        self.env = env
        self.d = env.drone
        self.log = []
        self.checks = []          # every candidate: far sighting -> close look -> Spark's answer
        self.map = json.load(open(MAP_FILE)) if os.path.exists(MAP_FILE) else {}

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        self.env.say(f"{action} {arg}")
        try:
            import muse_eye
            muse_eye.on_action(action, arg, result)       # Muse Brain panel: lit program line + live feed
        except Exception:
            pass
        try:                                       # the robot says the important steps out loud
            from speech import narrate, say
            line = narrate(action, arg, result)
            if line:
                say(line)
        except Exception:
            pass
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")

    def _save_map(self):
        json.dump(self.map, open(MAP_FILE, "w"), indent=2)

    # ---------- ground robot ----------
    def places(self):
        return list(self.env.locations)

    def go_to(self, place):
        ok, msg = self.env.go_to(str(place))
        self._record("go_to", place, msg)
        return ok

    def pick(self, item):
        item = str(item).strip()
        if not item or "unknown" in item.lower():
            self._record("pick", item, "REFUSED: not a valid item name")
            return False
        ok, msg = self.env.pick_seen(item)
        if ok or msg.startswith("could not see"):     # picked, or our own camera says it is not here
            self.map.pop(item, None)
            self._save_map()
        self._record("pick", item, msg)
        return ok

    def place_on(self, thing):
        env = self.env
        if not env.held:
            self._record("place_on", thing, "not holding anything")
            return False
        found = env.perceive(str(thing))
        if not found:
            self._record("place_on", thing, f"could not see '{thing}'")
            return False
        x, y, ztop = found[0]["pos"]
        # if something already lies there (e.g. an item in the tray), use the freest spot nearby
        others = [self._center(n) for n in env.objects
                  if n not in env.static and n != env.held[0] and n != "drone"]

        def room(q):
            return min([math.hypot(q[0] - o[0], q[1] - o[1]) for o in others] + [0.25])
        if "tray" in str(thing):                 # a tray holds several items: give each its own free part
            x, y = self._tray_spot(x, y)
        elif room((x, y)) < 0.1:
            x, y = max([(x, y), (x - 0.07, y), (x + 0.07, y), (x, y - 0.07), (x, y + 0.07)], key=room)
        ok, msg = self._place_gently(x, y, ztop)
        self._record("place_on", thing, msg)
        return ok

    def _tray_spot(self, x, y):
        """Where in the tray (center x, y) the held item fits best: it must stay inside the rim, and
        its outline should overlap the items already in the tray as little as possible."""
        import pybullet as p
        env = self.env

        def half(name):
            lo, hi = p.getAABB(env.objects[name])
            return (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
        hx, hy = half(env.held[0])
        inner = 0.13                                      # the tray's inside, from its middle to the rim
        near = [(self._center(n), half(n)) for n in env.objects
                if n not in env.static and n not in (env.held[0], "drone", "tray")
                and math.hypot(self._center(n)[0] - x, self._center(n)[1] - y) < 0.3]

        def keep_in(c, q, h):
            return c if h >= inner else min(max(q, c - inner + h), c + inner - h)

        def gap(q):                                       # > 0: no overlap with anything; < 0: overlap
            return min([max(abs(q[0] - o[0]) - (hx + oh[0]), abs(q[1] - o[1]) - (hy + oh[1]))
                        for o, oh in near] + [0.25])
        cands = [(keep_in(x, x + dx, hx), keep_in(y, y + dy, hy))
                 for dx in (0.0, -0.07, 0.07, -0.035, 0.035) for dy in (0.0, -0.06, 0.06, -0.03, 0.03)]
        return max(cands, key=gap)                        # ties keep the first (the middle)

    def _place_gently(self, x, y, ztop):
        """Lower the held item until it touches something, let the swing die out, then release."""
        import pybullet as p
        from mobile_env import EE_LINK
        env = self.env
        name, c = env.held
        body = env.objects[name]
        # the hand holds the item off-center: shift the hand so the ITEM's center lands on (x, y)
        hx, hy, _ = env.hand_pos()
        cx, cy = self._center(name)
        x, y = x + (hx - cx), y + (hy - cy)
        start = ztop + 0.28                           # high enough for tall items like bottles
        env.move_arm_to([x, y, ztop + 0.36])          # arrive from above, so a hanging item hits nothing
        env.move_arm_to([x, y, start], steps=120)
        hz, touched = start, False
        while hz > ztop + 0.01:
            hz -= 0.003
            q = p.calculateInverseKinematics(env.panda, EE_LINK, [x, y, hz], env.down(), maxNumIterations=50)
            env.arm_q = list(q[:7])
            env.step(3)
            if any(cp[2] != env.panda for cp in p.getContactPoints(bodyA=body)):
                touched = True
                break
        env.step(40)
        p.removeConstraint(c)
        env.held = None
        env.finger = 0.04
        env.step(120)
        # retreat: straight up, then back above the robot's front, then home (never sweep the table)
        env.move_arm_to([x, y, hz + 0.2], steps=120)
        env.retreat_home()
        return True, f"placed {name}" + ("" if touched else " (no contact felt)")

    def _center(self, name):
        """Middle of an item's bounding box (its base point can be off-center, e.g. a banana)."""
        import pybullet as p
        lo, hi = p.getAABB(self.env.objects[name])
        return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)

    def _table_spots(self):
        """Places to put an item down on the table in front: the spare spots first, then a grid."""
        env = self.env
        spots = list(env.free_spots.get(env.at, []))
        centers = getattr(env, "station_centers", {})
        if env.at in centers:
            cx, cy = centers[env.at]
            yaw = env.station_yaws[env.at]
            dx, dy = math.cos(yaw), math.sin(yaw)
            edge = 0.25 if abs(dx) > 0.5 else 0.30
            top = spots[0][2] if spots else 0.45
            for depth in (0.17, 0.12, 0.07):
                for k in (0.0, -0.08, 0.08, -0.15, 0.15):
                    d = edge - depth
                    spots.append((cx - dx * d - dy * k, cy - dy * d + dx * k, top))
        return spots

    def put_aside(self, side=None):
        env = self.env
        side = str(side).lower().strip() if side else None
        if not env.held:
            self._record("put_aside", side or "", "not holding anything")
            return False
        if side not in (None, "left", "middle", "right"):
            self._record("put_aside", side, "REFUSED: side must be 'left', 'middle' or 'right'")
            return False
        others = [self._center(n) for n in env.objects
                  if n not in env.static and n != env.held[0] and n != "drone"]
        for sx, sy, sz in self._table_spots():
            if side and table_side(env, env.at, sx, sy) != side:
                continue
            if all(math.hypot(sx - q[0], sy - q[1]) > 0.12 for q in others):
                ok, msg = self._place_gently(sx, sy, sz)
                self._record("put_aside", side or "", msg)
                return ok
        self._record("put_aside", side or "", f"no free spot {'on the ' + side + ' side ' if side else ''}here")
        return False

    def sees(self, item):
        item = str(item)
        ok = bool(self.env.perceive(item))
        self._record("sees", item, ok)
        return ok

    def is_holding(self):
        return self.env.held is not None

    def start_driving(self):
        if self.env.at != "packing station":
            self.env.go_to("packing station")        # a clear lane to drive in
        self.env.turn_to(0.0)
        self.env.set_wheels(1.2, 1.2)
        self.env.follow_camera = True
        self._record("start_driving", "", "the ground robot is driving east (+x) at a constant slow speed")
        return True

    def ground_robot_position(self):
        x, y, _ = self.env.pose()
        return [round(x, 2), round(y, 2)]

    # ---------- shared map ----------
    def recall(self, item):
        return self.map.get(str(item))

    def known_map(self):
        return dict(self.map)

    # ---------- drone ----------
    def _close_check(self, item, xyz, place, far_phrase, catalog):
        """Fly closer, find the object again with SAM 3.1, and let Muse Spark name the product."""
        import numpy as np
        from PIL import Image, ImageDraw
        from llm import spark_vision
        from mobile_env import ALIASES
        from perception import sam_find
        yaw = self.env.station_yaws[place]
        x, y, z = xyz
        self.d.fly_to(x - 0.45 * math.cos(yaw), y - 0.45 * math.sin(yaw), max(1.2, z + 0.7), 3.0)
        rgb, _ = self.d.camera_look([x, y, z])
        box, phrase = None, None
        for ph in ALIASES.get(item, [item]):
            boxes = sam_find(rgb, ph)
            if boxes:
                box = max(boxes, key=lambda b: b["conf"] or 0)["box"]
                phrase = ph
                break
        check = {"place": place, "item": item, "far_sam": far_phrase, "close_sam": phrase,
                 "spark": None, "kept": False}
        self.checks.append(check)
        if box is None:
            return False
        img = Image.fromarray(rgb).copy()
        x0, y0, x1, y1 = [float(v) for v in box]
        ImageDraw.Draw(img).rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(0, 255, 0), width=3)
        os.makedirs("runs", exist_ok=True)
        img.save(f"runs/check_{place.replace(' ', '_')}_{item.replace(' ', '_')}.png")
        q = ("Close-up image from a warehouse drone camera in a simple 3D simulation (objects look "
             "plain, so judge by shape and color). Which product is inside the green rectangle? "
             f"Choose exactly one from this product list: {', '.join(catalog)}, other. "
             "Reply with the product name only.")
        try:
            ans = spark_vision(q, np.array(img)).strip().lower().strip("*\"'. ")
        except Exception:
            ans = item + " (check unavailable)"
        check["spark"] = ans[:30]
        check["kept"] = item in ans
        try:
            import muse_eye
            muse_eye.log("drone", phrase, rgb, [{"box": box, "conf": None}], spark=ans[:30],
                         model="SAM 3.1 close look")
        except Exception:
            pass
        return check["kept"]

    def drone_scan(self, place, items):
        from mission_tasks import YCB_FILES
        from mobile_env import ALIASES
        from perception import sam_find
        place = str(place)
        if place not in self.env.station_centers:
            self._record("drone_scan", place, f"REFUSED: unknown place. Known: {list(self.env.station_centers)}")
            return []
        items = [str(it) for it in list(items)[:6]]
        catalog = sorted(set(YCB_FILES) | set(items))          # the warehouse's product list
        cx, cy = self.env.station_centers[place]
        yaw = self.env.station_yaws[place]
        # 1) far look: hover 1.1 m in front of the table; any name, may include false sightings
        h = getattr(self.env, "station_heights", {}).get(place, 0.45)    # table height (a high shelf is taller)
        self.d.fly_to(cx - 1.1 * math.cos(yaw), cy - 1.1 * math.sin(yaw), 1.4 + (h - 0.45), 3.0)
        rgb, depth = self.d.camera_look([cx, cy, h + 0.05])
        candidates, far_dets = [], []
        found = sam_many(rgb, items)                   # all items asked IN PARALLEL (much faster)
        for it in items:
            if found[it]:
                b, ph = found[it]
                x, y, z = self.d._pixel_to_world(*b["center"], depth)
                x = min(max(x, cx - 0.25), cx + 0.25)          # keep the point on the table
                y = min(max(y, cy - 0.30), cy + 0.30)
                candidates.append((it, [x, y, max(z, h + 0.02)], ph))
                far_dets.append(dict(b, label=it))
        try:
            import muse_eye
            muse_eye.log("drone", f"scan {place}", rgb, far_dets, chosen=-1, model="SAM 3.1 far look")
        except Exception:
            pass
        # 1b) small items (a strawberry) are only a few pixels from far away: look again from closer
        missing = [it for it in items if it not in [c[0] for c in candidates]]
        if missing:
            self.d.fly_to(cx - 0.6 * math.cos(yaw), cy - 0.6 * math.sin(yaw), h + 0.65, 3.0)
            rgb, depth = self.d.camera_look([cx, cy, h + 0.05])
            found = sam_many(rgb, missing)                # closer look (all names: SAM can miss "clamp")
            for it in missing:
                if found[it]:
                    b, ph = found[it]
                    x, y, z = self.d._pixel_to_world(*b["center"], depth)
                    x = min(max(x, cx - 0.25), cx + 0.25)
                    y = min(max(y, cy - 0.30), cy + 0.30)
                    candidates.append((it, [x, y, max(z, h + 0.02)], ph))
        # 2) close look for every candidate: SAM finds it again, Spark names the product
        seen, rejected = [], []
        for it, xyz, ph in candidates:
            (seen if self._close_check(it, xyz, place, ph, catalog) else rejected).append(it)
        for it in items:                       # forget old sightings that are no longer true
            if self.map.get(it) == place and it not in seen:
                self.map.pop(it)
        for it in seen:
            self.map[it] = place
        self._save_map()
        note = f"; rejected after a closer look: {rejected}" if rejected else ""
        self._record("drone_scan", place, f"seen: {seen}{note}")
        self._park_drone()                     # out of the arm's way before the robot comes
        return seen

    def _park_drone(self):
        """Fly up first (never sideways through the robot), then over to the parking spot."""
        x, y, z = self.d.position()
        if math.dist((x, y, z), DRONE_PARK) < 0.3:
            return
        self.d.fly_to(x, y, max(z, DRONE_PARK[2]), 1.5)
        self.d.fly_to(*DRONE_PARK, 3.0)

    def drone_position(self):
        return self.d.position()

    def drone_fly_to(self, x, y, z):
        x, y, z = float(x), float(y), float(z)
        if not 0.5 <= z <= 2.0:
            self._record("drone_fly_to", [x, y, z], "REFUSED: height must be 0.5-2.0 m")
            return self.d.position()
        pos = self.d.fly_to(x, y, z, 3.0)
        self._record("drone_fly_to", f"target [{x:.2f}, {y:.2f}, {z:.2f}]", f"after 3 s the drone is at {pos}")
        return pos

    def drone_find_pad(self):
        r = self.d.find("landing pad")
        xy = [r[0], r[1]] if r else None
        self._record("drone_find_pad", "", xy if xy else "pad not in view")
        return xy

    def drone_wait(self, seconds):
        s = min(max(float(seconds), 0.0), 5.0)
        self.env.step(int(s * 240))
        self._record("drone_wait", s, f"drone at {self.d.position()}")
        return True

    def drone_set_wind_correction(self, ax, ay):
        ax = max(-3.0, min(3.0, float(ax)))
        ay = max(-3.0, min(3.0, float(ay)))
        self.d.set_feedforward(ax, ay)
        self._record("drone_set_wind_correction", [round(ax, 3), round(ay, 3)], "ok")
        return True

    def drone_land_at(self, x, y):
        ok = self.d.land_at(float(x), float(y))
        self._record("drone_land_at", f"[{float(x):.2f}, {float(y):.2f}]",
                     ("LANDED on the pad" if ok else "MISSED the pad")
                     + f"; touched down at {self.d.touchdown_xy}")
        if ok:
            self.env.say(f"LANDED! touchdown {self.d.touchdown_miss * 100:.0f} cm from pad center")
        return ok

    def drone_fly_over(self, place):
        place = str(place)
        if place not in self.env.station_centers:
            self._record("drone_fly_over", place, f"REFUSED: unknown place. Known: {list(self.env.station_centers)}")
            return self.d.position()
        cx, cy = self.env.station_centers[place]
        pos = self.d.fly_to(cx, cy, 2.0, 3.0)
        self._record("drone_fly_over", place, f"drone at {pos}")
        return pos

    def drone_grab(self, item):
        ok, msg = self.d.grab(str(item))
        self._record("drone_grab", item, msg)
        return ok

    def handoff_to_robot(self, place):
        import pybullet as p
        env, place = self.env, str(place)
        if not getattr(env, "basket", False):
            self._record("handoff_to_robot", place, "REFUSED: this robot has no basket")
            return False
        if not self.d.cargo:
            self._record("handoff_to_robot", place, "the drone is not carrying anything")
            return False
        if place not in env.locations:
            self._record("handoff_to_robot", place, f"REFUSED: unknown place. Known: {list(env.locations)}")
            return False
        name = self.d.cargo[0]
        tx, ty, _ = env.locations[place]
        px, py, _ = env.pose()
        if math.hypot(tx - px, ty - py) < 0.7:          # too close for a drive: back off to the free middle
            env.navigate_to(0.0, 0.0, tol=0.08)
            px, py, _ = env.pose()
        env.turn_to(math.atan2(ty - py, tx - px))
        bx, by, _ = env.basket_pos()
        high = env.pad_top + 0.95
        dz = self.d.position()[2]
        self.d.fly_to(self.d.position()[0], self.d.position()[1], max(dz, high), 1.0)
        self.d.fly_to(bx, by, high, 3.0)                # the drone waits above the basket
        env.set_wheels(1.2, 1.2)                        # the robot drives slowly toward the place
        env.step(120)                                   # 0.5 s to reach a steady speed
        vx, vy = p.getBaseVelocity(env.husky)[0][:2]    # the robot reports its speed (wheel odometry)
        ok, msg, info = self.d.drop_on_moving(vx, vy)
        env.step(120)
        env.stop()
        found = env.look_in_basket(name)                # the basket camera checks with SAM 3.1
        self.handoff_info = dict(info, seen_in_basket=bool(found))
        self._park_drone()
        env.go_to(place)
        if ok and found:
            self._record("handoff_to_robot", place, f"{msg}; the basket camera sees the {name}; robot parked at {place}")
            return True
        why = msg if not ok else f"{msg}, but the basket camera does not see the {name} (it may have bounced out)"
        self._record("handoff_to_robot", place, f"FAILED: {why}")
        return False

    def land_on_moving_robot(self):
        """Composite skill (the tested 'land' job): the robot drives off, the drone measures the pad's speed
        with SAM 3.1 and lands on it with the RL reflex."""
        from skill_tasks import plan_land
        if self.d.latched:
            self._record("land_on_moving_robot", "", "already landed")
            return True
        ok = bool(plan_land(self))
        self._record("land_on_moving_robot", "", "LANDED on the moving robot" if ok else "missed the pad")
        return ok

    def take_from_basket(self, item):
        env, item = self.env, str(item)
        if env.held:
            self._record("take_from_basket", item, f"already holding {env.held[0]}")
            return False
        found = env.look_in_basket(item)
        if not found:
            self._record("take_from_basket", item, f"the basket camera does not see the {item}")
            return False
        ok, msg = env.pick_from_back(found["grasp"])
        if not ok and msg.startswith("the fingers could not close"):
            # like pick(): look again (the touch may have moved it) and try with the hand turned 90 degrees
            again = env.look_in_basket(item)
            if again:
                env.grip_yaw = math.pi / 2
                ok, msg = env.pick_from_back(again["grasp"])
                env.grip_yaw = 0.0
        self._record("take_from_basket", item, msg)
        return ok

    def drone_drop_into(self, thing):
        ok, msg = self.d.drop_into(str(thing))
        self._record("drone_drop_into", thing, msg)
        self._park_drone()
        return ok

    def drone_land_rl(self, vx, vy):
        ok = self.d.land_rl(float(vx), float(vy))
        self._record("drone_land_rl", f"[{float(vx):.3f}, {float(vy):.3f}]",
                     ("LANDED on the pad" if ok else "MISSED the pad")
                     + f"; touched down at {self.d.touchdown_xy}; clean pad sightings: {len(self.d.track_log)}"
                     + f"; fitted pad velocity: {self.d.track_velocity}")
        if ok:
            self.env.say(f"LANDED (RL reflex)! touchdown {self.d.touchdown_miss * 100:.0f} cm from pad center")
        return ok

    def drone_land_tracking(self, vx, vy):
        ok = self.d.land_tracking(float(vx), float(vy))
        self._record("drone_land_tracking", f"[{float(vx):.3f}, {float(vy):.3f}]",
                     ("LANDED on the pad" if ok else "MISSED the pad")
                     + f"; touched down at {self.d.touchdown_xy}; clean pad sightings: {len(self.d.track_log)}"
                     + f"; fitted pad velocity: {self.d.track_velocity}")
        if ok:
            self.env.say(f"LANDED! touchdown {self.d.touchdown_miss * 100:.0f} cm from pad center")
        return ok
