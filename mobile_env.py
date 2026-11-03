import math
import os
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

HOME = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]
LOOK = [0, -1.3, 0, -2.5, 0, 1.4, 0.8]     # arm pulled up and back: the front camera sees the whole table
EE_LINK = 11
DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
W, H = 640, 480
ALIASES = {"tray": ["green box", "green tray", "green container"]}
TABLE_COLOR = (0.3, 0.3, 0.32, 1)


def square_box(body, station_yaw, x, y):
    """Put a box-shaped item square to the table edge, narrow side across the gripper's fingers (a person
    stocking a shelf would do the same): try turns of 0-180 degrees, keep the smallest footprint, then the
    turn whose width across the fingers is smallest. The item keeps its spot (x, y)."""
    pos, orn0 = p.getBasePositionAndOrientation(body)
    across_y = abs(math.cos(station_yaw)) > 0.5          # robot faces along x: the fingers close along y
    best = None
    for deg in range(0, 180, 5):
        orn = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, math.radians(deg)]), [0, 0, 0], orn0)[1]
        p.resetBasePositionAndOrientation(body, pos, orn)
        lo, hi = p.getAABB(body)
        w, d = hi[0] - lo[0], hi[1] - lo[1]
        across = d if across_y else w
        key = (round(w * d, 4), across)
        if best is None or key < best[0]:
            best = (key, orn)
    p.resetBasePositionAndOrientation(body, pos, best[1])
    lo, hi = p.getAABB(body)
    p.resetBasePositionAndOrientation(body, [pos[0] + x - (lo[0] + hi[0]) / 2, pos[1] + y - (lo[1] + hi[1]) / 2,
                                             pos[2]], best[1])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MobileEnv:
    def __init__(self, gui=True):
        self.gui = gui
        if gui:                                   # big window, soft sky-grey background
            p.connect(p.GUI, options="--width=1600 --height=900 --background_color_red=0.80 "
                                     "--background_color_green=0.84 --background_color_blue=0.88")
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # ---- warehouse look: concrete floor, yellow safety lines (or the home look with --home) ----
        import home
        if home.active():
            home.build_room(p)
        else:
            self._warehouse_floor()
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            try:                                  # sharper shadows from a high light (newer PyBullet)
                p.configureDebugVisualizer(lightPosition=[2.5, -3.0, 6.0], shadowMapResolution=4096,
                                           shadowMapWorldSize=12, shadowMapIntensity=0.45)
            except Exception:
                pass
            p.resetDebugVisualizerCamera(cameraDistance=3.2, cameraYaw=35, cameraPitch=-30,
                                         cameraTargetPosition=[0.5, 0.0, 0.3])

        self.objects = {}
        self.hooks = []           # extra controllers (drone, pad) run every physics step
        self.static = set()       # tables, trays, pads: never grasped
        self.held = None
        self.finger = 0.04
        self.locations = {}
        self.free_spots = {}
        self.at = None
        self.odometer = 0.0
        self.main_view = "overview"
        self.follow_camera = False
        self.slowmo = 0.0 if os.environ.get("SKILLFORGE_FAST") == "1" else 1.0   # 0 = no real-time wait (--fast)
        self.views = {}           # extra named cameras: name -> (eye, target)
        self.slip_prob = 0.0      # HIDDEN fault: chance that a grasp slips while lifting
        self.fault_rng = random.Random(0)
        self._label = None
        self.basket = False       # True when the pad is a basket that carries items (hand-off scene)
        self.verify = True        # ask Muse Spark to confirm the item before every grasp

        # ---- Husky: let it land first, then measure the top plate ----
        self.husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
        self.wheels = {}
        plate = None
        for j in range(p.getNumJoints(self.husky)):
            info = p.getJointInfo(self.husky, j)
            if "wheel" in info[1].decode():
                self.wheels[info[1].decode()] = j
            if info[12].decode() == "top_plate_link":
                plate = j
        for _ in range(240):
            p.stepSimulation()
        base_pos, _ = p.getBasePositionAndOrientation(self.husky)
        top_z = p.getAABB(self.husky, plate)[1][2]
        self.top_z = top_z
        self.base_z0 = base_pos[2]

        # ---- Panda arm mounted on the top plate ----
        self.panda = p.loadURDF("franka_panda/panda.urdf",
                                [base_pos[0] + 0.1, base_pos[1], top_z], useFixedBase=False)
        for j, q in enumerate(HOME):
            p.resetJointState(self.panda, j, q)
        inertial_z = p.getDynamicsInfo(self.panda, -1)[3][2]
        mount = p.createConstraint(self.husky, -1, self.panda, -1, p.JOINT_FIXED, [0, 0, 0],
                                   [0.1, 0, top_z - base_pos[2] + inertial_z], [0, 0, 0])
        p.changeConstraint(mount, maxForce=5000)
        for hl in range(-1, p.getNumJoints(self.husky)):
            for pl in (-1, 0):
                p.setCollisionFilterPair(self.husky, self.panda, hl, pl, 0)
        self.arm_q = list(HOME)
        self.step(120)
        self.control_room = None
        if gui and os.environ.get("SKILLFORGE_NO_ROOM") != "1":
            try:
                from muse_eye import ControlRoom
                fast = os.environ.get("SKILLFORGE_FAST") == "1"
                self.control_room = ControlRoom(self, every=240 if fast else 60)   # --fast: 1 picture/s, not 4
            except Exception as e:
                print("control room off:", e)

    # ---------- world building ----------
    def _warehouse_floor(self):
        plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0])
        floor_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[8, 8, 0.002],
                                      rgbaColor=[0.62, 0.62, 0.6, 1])
        p.createMultiBody(0, -1, floor_v, [0, 0, -0.002])
        line_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.0, 0.03, 0.001],
                                     rgbaColor=[0.95, 0.8, 0.1, 1])
        for y in (-2.4, 2.4):
            p.createMultiBody(0, -1, line_v, [0, y, 0.001])
        self._decorate()

    def _decorate(self):
        """Warehouse look (visual only, no collisions): walls with a dark baseboard, a floor-tile grid,
        blue work-zone squares around the middle. Kept plain so the cameras still see items clearly."""
        wall = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.8, 0.05, 1.3], rgbaColor=[0.86, 0.88, 0.9, 1])
        wall_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 4.8, 1.3], rgbaColor=[0.82, 0.85, 0.88, 1])
        base = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.8, 0.06, 0.12], rgbaColor=[0.2, 0.22, 0.26, 1])
        base_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 4.8, 0.12], rgbaColor=[0.2, 0.22, 0.26, 1])
        stripe = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.8, 0.06, 0.08], rgbaColor=[0.1, 0.35, 0.75, 1])
        stripe_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 4.8, 0.08], rgbaColor=[0.1, 0.35, 0.75, 1])
        for y in (-4.8, 4.8):
            p.createMultiBody(0, -1, wall, [0, y, 1.3])
            p.createMultiBody(0, -1, base, [0, y * 0.995, 0.12])
            p.createMultiBody(0, -1, stripe, [0, y * 0.995, 1.2])
        for x in (-4.8, 4.8):
            p.createMultiBody(0, -1, wall_s, [x, 0, 1.3])
            p.createMultiBody(0, -1, base_s, [x * 0.995, 0, 0.12])
            p.createMultiBody(0, -1, stripe_s, [x * 0.995, 0, 1.2])
        grid = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.75, 0.004, 0.0005], rgbaColor=[0.55, 0.55, 0.54, 1])
        grid_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.004, 4.75, 0.0005], rgbaColor=[0.55, 0.55, 0.54, 1])
        for k in range(-4, 5):
            p.createMultiBody(0, -1, grid, [0, k * 1.0, 0.0003])
            p.createMultiBody(0, -1, grid_s, [k * 1.0, 0, 0.0003])
        zone = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.05, 0.02, 0.0008], rgbaColor=[0.15, 0.45, 0.85, 1])
        zone_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 1.05, 0.0008], rgbaColor=[0.15, 0.45, 0.85, 1])
        for v in (-1.05, 1.05):
            p.createMultiBody(0, -1, zone, [0, v, 0.0006])
            p.createMultiBody(0, -1, zone_s, [v, 0, 0.0006])

    def label_places(self):
        """Floating names above every place (GUI only; cameras do not see them)."""
        if not self.gui:
            return
        for name, (x, y, yaw) in self.locations.items():
            cx, cy = x + 0.85 * math.cos(yaw), y + 0.85 * math.sin(yaw)
            import home
            p.addUserDebugText(home.label(name).upper(), [cx, cy, 1.05], textColorRGB=[0.05, 0.2, 0.55], textSize=1.4)

    def add_table(self, name, center, size=(0.5, 0.6), height=0.45, rgba=TABLE_COLOR, yaw=0.0):
        half = [size[0] / 2, size[1] / 2, height / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        import home
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=home.table_color(name, rgba))
        self.objects[name] = p.createMultiBody(0, col, vis, [center[0], center[1], height / 2],
                                               p.getQuaternionFromEuler([0, 0, yaw]))
        self.static.add(name)
        return height

    def add_tray(self, name, center, z, size=0.2, wall=0.01, height=0.05, rgba=(0.1, 0.75, 0.2, 1)):
        cx, cy = center
        half = size / 2
        parts = [((half, half, wall / 2), (cx, cy, z + wall / 2)),
                 ((half, wall / 2, height / 2), (cx, cy + half, z + height / 2)),
                 ((half, wall / 2, height / 2), (cx, cy - half, z + height / 2)),
                 ((wall / 2, half, height / 2), (cx + half, cy, z + height / 2)),
                 ((wall / 2, half, height / 2), (cx - half, cy, z + height / 2))]
        ids = []
        for half_ext, pos in parts:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=rgba)
            ids.append(p.createMultiBody(0, col, vis, pos))
        self.objects[name] = ids[0]
        self.static.add(name)

    def add_cube(self, name, xy, z, rgba):
        body = p.loadURDF("cube_small.urdf", [xy[0], xy[1], z + 0.03])
        p.changeVisualShape(body, -1, rgbaColor=rgba)
        self.objects[name] = body

    def add_cube_on(self, name, below, rgba):
        x, y, z = self.get_pos(below)
        body = p.loadURDF("cube_small.urdf", [x, y, z + 0.051])
        p.changeVisualShape(body, -1, rgbaColor=rgba)
        self.objects[name] = body

    def add_location(self, name, pose, spots):
        self.locations[name] = pose
        self.free_spots[name] = spots

    def add_station(self, name, center, yaw, rgba=TABLE_COLOR):
        # a station facing a diagonal (home mode's storeroom) gets its table turned to face the robot squarely;
        # the usual stations (0, 90, 180, 270 degrees) keep their table exactly as before
        top = self.add_table(name, center, rgba=rgba, yaw=yaw if abs(math.sin(2 * yaw)) > 0.1 else 0.0)
        dx, dy = math.cos(yaw), math.sin(yaw)
        lx, ly = -dy, dx
        edge = 0.25 if abs(dx) > 0.5 else 0.30
        cx, cy = center
        front = (cx - dx * (edge - 0.07), cy - dy * (edge - 0.07))
        back = (cx - dx * (edge - 0.17), cy - dy * (edge - 0.17))
        items = [(front[0] + lx * k, front[1] + ly * k) for k in (-0.15, 0.0, 0.15)]
        spare = [(back[0] + lx * k, back[1] + ly * k, top) for k in (-0.15, 0.15)]
        self.add_location(name, (cx - dx * 0.85, cy - dy * 0.85, yaw), spare)
        return {"top": top, "items": items}

    # ---------- landing pad: on top of everything on the Husky's rear deck ----------
    def add_landing_pad(self, size=0.45, rgba=(1.0, 0.45, 0.0, 1), offset=-0.38, basket=False):
        """basket=True: the pad gets low walls and rides on the robot as a real body (fixed joint), so an
        item dropped into it by the drone travels with the robot (a hand-off on the move)."""
        deck = max(p.getAABB(self.husky, j)[1][2] for j in range(-1, p.getNumJoints(self.husky)))
        pad_z = deck + 0.03
        half = [size / 2, size / 2, 0.006]
        self.pad_top = pad_z + 0.006
        self.pad_half = size / 2
        self.pad_offset = [offset, 0.0, pad_z - self.base_z0]
        if basket:
            self._add_basket(size, rgba, half)
            return
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        self.pad = p.createMultiBody(0, col, vis, [0, 0, 1])
        for hl in range(-1, p.getNumJoints(self.husky)):
            p.setCollisionFilterPair(self.husky, self.pad, hl, -1, 0)
        for pl in range(-1, p.getNumJoints(self.panda)):
            p.setCollisionFilterPair(self.panda, self.pad, pl, -1, 0)
        self.objects["landing pad"] = self.pad
        self.static.add("landing pad")
        self._follow_pad()
        self.hooks.append(self._follow_pad)

    def _add_basket(self, size, rgba, half):
        wall_h, t = 0.045, 0.008
        h = size / 2
        boxes = [(half, [0, 0, 0])]                                        # the orange floor (the pad)
        for dx, dy, hx, hy in [(h - t, 0, t, h), (-(h - t), 0, t, h), (0, h - t, h, t), (0, -(h - t), h, t)]:
            boxes.append(([hx, hy, wall_h / 2], [dx, dy, 0.006 + wall_h / 2]))
        col = p.createCollisionShapeArray([p.GEOM_BOX] * len(boxes), halfExtents=[b[0] for b in boxes],
                                          collisionFramePositions=[b[1] for b in boxes])
        colors = [list(rgba)] + [[0.25, 0.25, 0.28, 1]] * 4
        vis = p.createVisualShapeArray([p.GEOM_BOX] * len(boxes), halfExtents=[b[0] for b in boxes],
                                       visualFramePositions=[b[1] for b in boxes], rgbaColors=colors)
        pos, orn = p.getBasePositionAndOrientation(self.husky)
        world, _ = p.multiplyTransforms(pos, orn, self.pad_offset, [0, 0, 0, 1])
        self.pad = p.createMultiBody(0.2, col, vis, world, orn)
        p.changeDynamics(self.pad, -1, lateralFriction=1.2)
        c = p.createConstraint(self.husky, -1, self.pad, -1, p.JOINT_FIXED, [0, 0, 0], self.pad_offset, [0, 0, 0])
        p.changeConstraint(c, maxForce=3000)
        for hl in range(-1, p.getNumJoints(self.husky)):
            p.setCollisionFilterPair(self.husky, self.pad, hl, -1, 0)
        for pl in range(-1, p.getNumJoints(self.panda)):
            p.setCollisionFilterPair(self.panda, self.pad, pl, -1, 0)
        self.objects["landing pad"] = self.pad
        self.static.add("landing pad")
        self.basket = True

    def basket_pos(self):
        return list(p.getBasePositionAndOrientation(self.pad)[0])

    def _follow_pad(self):
        pos, orn = p.getBasePositionAndOrientation(self.husky)
        world, _ = p.multiplyTransforms(pos, orn, self.pad_offset, [0, 0, 0, 1])
        yaw = p.getEulerFromQuaternion(orn)[2]
        p.resetBasePositionAndOrientation(self.pad, world, p.getQuaternionFromEuler([0, 0, yaw]))

    def get_pos(self, name):
        return list(p.getBasePositionAndOrientation(self.objects[name])[0])

    # ---------- basics ----------
    def step(self, n=1):
        self.t_steps = getattr(self, "t_steps", 0) + n
        for _ in range(n):
            for j, q in enumerate(self.arm_q):
                p.setJointMotorControl2(self.panda, j, p.POSITION_CONTROL, q, force=240)
            for j in (9, 10):
                p.setJointMotorControl2(self.panda, j, p.POSITION_CONTROL, self.finger, force=20)
            for hook in self.hooks:
                hook()
            p.stepSimulation()
            if self.gui:
                if self.follow_camera:
                    x, y, _ = self.pose()
                    p.resetDebugVisualizerCamera(3.0, 35, -30, [x, y, 0.3])
                if self.slowmo > 0:
                    time.sleep(self.slowmo / 240)

    def set_wheels(self, left, right):
        for name, j in self.wheels.items():
            v = left if "left" in name else right
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL, targetVelocity=v, force=100)

    def pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.husky)
        return pos[0], pos[1], p.getEulerFromQuaternion(orn)[2]

    def say(self, text):
        """Live label floating above the robot (GUI only)."""
        if not self.gui:
            return
        kw = {"replaceItemUniqueId": self._label} if self._label is not None else {}
        self._label = p.addUserDebugText(str(text)[:60], [0, 0, 1.7], textColorRGB=[0.1, 0.1, 0.1],
                                         textSize=1.5, parentObjectUniqueId=self.husky, **kw)

    def stop(self):
        self.set_wheels(0, 0)
        self.step(120)

    # ---------- feedback navigation ----------
    def turn_to(self, yaw, tol=0.03, max_steps=1500):
        for _ in range(max_steps):
            err = wrap(yaw - self.pose()[2])
            if abs(err) < tol:
                break
            w = max(-4.0, min(4.0, 6.0 * err))
            self.set_wheels(-w, w)
            self.step()
        self.stop()
        return abs(wrap(yaw - self.pose()[2]))

    def navigate_to(self, x, y, tol=0.03, max_steps=4000):
        for _ in range(max_steps):
            px, py, yaw = self.pose()
            dx, dy = x - px, y - py
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            if dist < 0.15:
                # final approach: drive straight along the heading (no turning on the spot,
                # because skid-steer turns make the robot slide sideways)
                along = dx * math.cos(yaw) + dy * math.sin(yaw)
                if abs(along) < tol:
                    break
                v = math.copysign(max(1.0, min(8.0, 6.0 * abs(along))), along)
                self.set_wheels(v, v)
            else:
                err = wrap(math.atan2(dy, dx) - yaw)
                if abs(err) > 0.3:
                    w = max(-4.0, min(4.0, 6.0 * err))
                    self.set_wheels(-w, w)
                else:
                    v = max(1.0, min(8.0, 6.0 * dist))
                    w = 6.0 * err
                    self.set_wheels(v - w, v + w)
            self.step()
            nx, ny, _ = self.pose()
            self.odometer += math.hypot(nx - px, ny - py)
        self.stop()
        px, py, _ = self.pose()
        return math.hypot(x - px, y - py)

    def go_to(self, name):
        if name not in self.locations:
            return False, f"unknown place '{name}'. Known places: {list(self.locations)}"
        x, y, yaw = self.locations[name]
        c, s = math.cos(yaw), math.sin(yaw)
        err = 1.0
        for _ in range(3):
            # stop 0.35 m short, face the table, then drive straight in (accurate parking)
            self.navigate_to(x - 0.35 * c, y - 0.35 * s, tol=0.05)
            self.turn_to(yaw)
            self.navigate_to(x, y)
            self.turn_to(yaw)
            px, py, _ = self.pose()
            err = math.hypot(x - px, y - py)
            if err < 0.05:
                break
        self.at = name
        return True, f"arrived at {name} (error {err * 100:.1f} cm)"

    # ---------- arm ----------
    def hand_pos(self):
        return list(p.getLinkState(self.panda, EE_LINK)[4])

    def down(self):
        """Gripper pointing down, turned with the robot (so the wrist never needs a half turn)."""
        return p.getQuaternionFromEuler([math.pi, 0, self.pose()[2] + getattr(self, "grip_yaw", 0.0)])

    def move_arm_to(self, target, steps=240):
        for _ in range(steps):
            q = p.calculateInverseKinematics(self.panda, EE_LINK, target, self.down(), maxNumIterations=50)
            self.arm_q = list(q[:7])
            self.step()
        return math.dist(self.hand_pos(), target)

    def retreat_home(self):
        """Go home without sweeping the table: first up and back above the robot's own front, then home."""
        x, y, yaw = self.pose()
        hx, hy, hz = self.hand_pos()
        self.move_arm_to([hx, hy, max(hz, 0.9)], steps=100)                       # straight up first
        self.move_arm_to([x + 0.35 * math.cos(yaw), y + 0.35 * math.sin(yaw), 0.95], steps=140)
        self.arm_home()

    def glide(self, target, steps=240):
        """Move the joints smoothly (small steps) instead of jumping: a jump can swing the arm through the table."""
        start = [p.getJointState(self.panda, j)[0] for j in range(7)]
        for k in range(1, steps + 1):
            a = k / steps
            self.arm_q = [s + a * (t - s) for s, t in zip(start, target)]
            self.step()
        self.step(60)                     # let the arm settle (a camera picture needs a still arm)

    def arm_home(self):
        self.glide(list(HOME), 240)

    def go_home(self):
        self.arm_home()

    def _closest_object_to_hand(self):
        hand = np.array(self.hand_pos())
        best, best_key = None, (1e9, 1e9)
        for n, body in self.objects.items():
            if n in self.static or n == "drone":
                continue
            lo, hi = (np.array(v) for v in p.getAABB(body))
            outline_d = np.linalg.norm(hand - np.clip(hand, lo, hi))
            center_d = np.linalg.norm(hand - np.array(self.get_pos(n)))
            if (outline_d, center_d) < best_key:
                best, best_key = n, (outline_d, center_d)
        return best, best_key[0]

    def pick_at(self, top_xyz):
        if self.held:
            return False, "already holding " + self.held[0]
        x, y, ztop = top_xyz
        z = ztop - 0.025
        self.finger = 0.04
        self.move_arm_to([x, y, z + 0.15])
        if self.slip_prob > 0 and self.fault_rng.random() < self.slip_prob:
            # the grip does not hold: the hand comes up empty and the item stays on the table
            self.move_arm_to([x, y, z + 0.06], steps=120)
            self.say("grasp slipped!")
            self.move_arm_to([x, y, z + 0.15], steps=120)
            self.retreat_home()
            return False, "grasp slipped: the gripper came up empty (the item is still on the table)"
        self.move_arm_to([x, y, z + 0.01])
        best, best_d = self._closest_object_to_hand()
        if best is None or best_d > 0.03:
            self.move_arm_to([x, y, z + 0.15])
            self.retreat_home()
            return False, f"nothing graspable under the hand (closest {best_d:.3f} m)"
        # PHYSICAL GRIP: both fingers must really touch the item, with the item between them
        ok, gap = self._close_and_check(best)
        turned = abs(getattr(self, "grip_yaw", 0.0)) > 0.1
        if not ok:
            self.finger = 0.04
            self.step(60)
            self.move_arm_to([x, y, z + 0.15])
            self.retreat_home()
            self.grip_yaw = 0.0
            return False, "the fingers could not close around it (no contact on both fingers)"
        h = p.getLinkState(self.panda, EE_LINK)
        inv_pos, inv_orn = p.invertTransform(h[4], h[5])
        o_pos, o_orn = p.getBasePositionAndOrientation(self.objects[best])
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, o_pos, o_orn)
        # the item is held by the closed fingers; a fixed joint keeps it stable, but it BREAKS above 60 N
        c = p.createConstraint(self.panda, EE_LINK, self.objects[best], -1, p.JOINT_FIXED,
                               [0, 0, 0], rel_pos, [0, 0, 0], rel_orn)
        p.changeConstraint(c, maxForce=60)
        self.held = (best, c)
        self.move_arm_to([x, y, z + 0.15])
        self.retreat_home()          # up first: the held item must not sweep the other items off the table
        self.grip_yaw = 0.0
        return True, f"picked object (two-finger grip, {gap * 1000:.0f} mm between the fingers" + \
            (", hand turned 90 degrees)" if turned else ")")

    def _close_and_check(self, name):
        """Close the fingers, then measure: contact on BOTH finger pads and the item really between them."""
        self.finger = 0.0
        self.step(60)
        b = self.objects[name]
        c9 = p.getContactPoints(self.panda, b, 9, -1)
        c10 = p.getContactPoints(self.panda, b, 10, -1)
        gap = p.getJointState(self.panda, 9)[0] + p.getJointState(self.panda, 10)[0]
        return bool(c9) and bool(c10) and gap > 0.004, gap

    def place_at(self, xy, ztop):
        if not self.held:
            return False, "not holding anything"
        name = self.held[0]
        zc = ztop + 0.03
        self.move_arm_to([xy[0], xy[1], zc + 0.12])
        self.move_arm_to([xy[0], xy[1], zc + 0.03])
        p.removeConstraint(self.held[1])
        self.held = None
        self.finger = 0.04
        self.step(90)
        self.move_arm_to([xy[0], xy[1], zc + 0.12])
        self.retreat_home()
        return True, "placed " + name

    def put_aside(self):
        if not self.held:
            return False, "not holding anything"
        others = [self.get_pos(n) for n in self.objects
                  if n not in self.static and n != self.held[0] and n != "drone"]
        for sx, sy, sz in self.free_spots.get(self.at, []):
            if all(math.hypot(sx - q[0], sy - q[1]) > 0.09 for q in others):
                return self.place_at([sx, sy], sz)
        return False, "no free spot here"

    # ---------- cameras ----------
    def camera(self, view="front"):
        if view in self.views:
            eye, target = self.views[view]
            self._view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
        elif view == "overview":
            self._view = p.computeViewMatrix([0.8, -0.6, 3.2], [0.8, -0.6, 0.0], [1, 0, 0])
        elif view == "station":
            self._view = p.computeViewMatrix([0.0, -2.35, 1.35], [0.0, -1.42, 0.45], [0, 0, 1])
        elif view.startswith("table:"):           # close-up of one table (for the goal checks), e.g. "table:station A"
            place = view[6:]
            cx, cy = self.station_centers[place]
            yaw = self.station_yaws[place]
            c, s = math.cos(yaw), math.sin(yaw)
            h = getattr(self, "station_heights", {}).get(place, 0.45)
            # from the side and a little in front: the parked robot (in front of the table) does not block it
            self._view = p.computeViewMatrix([cx - 0.45 * c + 0.95 * s, cy - 0.45 * s - 0.95 * c, h + 0.9],
                                             [cx, cy, h], [0, 0, 1])
        elif view == "basket":                    # a small camera on a mast, looking down into the basket
            bx, by, bz = self.basket_pos()
            x, y, yaw = self.pose()
            c, s = math.cos(yaw), math.sin(yaw)
            self._view = p.computeViewMatrix([bx + 0.22 * c, by + 0.22 * s, bz + 0.62], [bx, by, bz], [-c, -s, 0])
        elif view == "landing":
            x, y, _ = self.pose()
            self._view = p.computeViewMatrix([x - 1.6, y - 1.6, 1.7], [x, y, 0.4], [0, 0, 1])
        else:
            x, y, yaw = self.pose()
            c, s = math.cos(yaw), math.sin(yaw)
            eye = [x + 0.5 * c, y + 0.5 * s, 1.0]
            target = [x + 0.85 * c, y + 0.85 * s, 0.45]       # tilted down: the front row stays in view
            self._view = p.computeViewMatrix(eye, target, [0, 0, 1])
        self._proj = p.computeProjectionMatrixFOV(60, W / H, 0.1, 6.0)
        _, _, rgb, depth, _ = p.getCameraImage(W, H, self._view, self._proj,
                                               renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        return rgb, np.reshape(depth, (H, W))

    def _matrix(self):
        V = np.array(self._view).reshape(4, 4, order="F")
        P = np.array(self._proj).reshape(4, 4, order="F")
        return P @ V

    def pixel_to_world(self, u, v, depth):
        d = depth[int(v), int(u)]
        ndc = np.array([2 * u / W - 1, 1 - 2 * v / H, 2 * d - 1, 1.0])
        w = np.linalg.inv(self._matrix()) @ ndc
        return (w[:3] / w[3]).tolist()

    def clear_view(self):
        """Pull the arm up and back (a fixed joint pose) so it does not block the front camera."""
        self.glide(list(LOOK), 120)

    def _box_points(self, box, center, depth):
        """3D points sampled inside a detection box: (pixel distance to the center, [x, y, z])."""
        x0, y0, x1, y1 = [float(v) for v in box]
        pts = [(0.0, self.pixel_to_world(*center, depth))]
        for fy in np.linspace(0.1, 0.9, 9):
            for fx in np.linspace(0.1, 0.9, 9):
                u, v = min(x0 + fx * (x1 - x0), W - 1), min(y0 + fy * (y1 - y0), H - 1)
                pts.append((math.hypot(u - center[0], v - center[1]), self.pixel_to_world(u, v, depth)))
        return [t for t in pts if t[1][2] > 0.3]           # drop rays that hit the floor

    def _grasp_point(self, box, center, depth):
        """A point on the TOP of the object, where the hand grasps it. The box center can miss a
        curved object (a banana) or hit the side of a tall one (a bottle), so: keep the points that
        stand up from the table, then take the top layer, nearest to the box center."""
        pts = self._box_points(box, center, depth)
        if not pts:
            return self.pixel_to_world(*center, depth)
        zmin = min(w[2] for _, w in pts)
        on_object = [t for t in pts if t[1][2] > zmin + 0.012] or pts
        ztop = max(w[2] for _, w in on_object)
        top = [t for t in on_object if t[1][2] > ztop - 0.015]
        return min(top, key=lambda t: t[0])[1]

    def _surface_point(self, box, center, depth):
        """The 3D point at the box center (for placing ON / INTO things such as the tray)."""
        pos = self.pixel_to_world(*center, depth)
        if pos[2] > 0.3:
            return pos
        pts = self._box_points(box, center, depth)
        return min(pts, key=lambda t: t[0])[1] if pts else pos

    def perceive(self, phrase):
        # reuse the last answer when nothing changed (e.g. sees("banana") then pick("banana") at once):
        # same item, robot not moved, nothing grasped or released, less than 8 s ago -> no new SAM/Spark call
        x, y, yaw = self.pose()
        key = (phrase, round(x, 2), round(y, 2), round(yaw, 2), self.held[0] if self.held else None)
        c = getattr(self, "_pcache", None)
        if c and c[0] == key and getattr(self, "t_steps", 0) - c[1] < 1920:
            return [dict(f) for f in c[2]]
        found = self._perceive(phrase)
        self._pcache = (key, getattr(self, "t_steps", 0), [dict(f) for f in found])
        return found

    def _perceive(self, phrase):
        from perception import sam_find
        self.clear_view()
        rgb, depth = self.camera("front")
        candidates = ALIASES.get(phrase, [phrase])
        if "cube" in phrase:
            candidates = [phrase, phrase.replace("cube", "block"), phrase.replace("cube", "box")]
        boxes = []
        for ph in candidates:
            boxes = sam_find(rgb, ph)
            if boxes:
                break
        found = []
        for b in sorted(boxes, key=lambda b: -(b["conf"] or 0)):
            found.append({"pos": self._surface_point(b["box"], b["center"], depth),
                          "grasp": self._grasp_point(b["box"], b["center"], depth),
                          "conf": b["conf"], "box": b["box"]})
        found, spark = self._confirm(rgb, phrase, found)
        self.last_view = (rgb, phrase, boxes)
        try:
            import muse_eye
            muse_eye.log("front", phrase, rgb, found or boxes, chosen=0,
                         xyz=(found[0]["grasp"] if found else None), spark=spark)
        except Exception:
            pass
        return found

    def _confirm(self, rgb, phrase, found):
        """Ask Muse Spark before acting: which product is in the best SAM box? If Spark names a DIFFERENT
        known product, try the next box (up to 3). Tables, trays and pads are not checked."""
        if not found or phrase not in self.objects or phrase in self.static or not self.verify:
            return found, None
        import muse_eye
        names = [phrase] + ALIASES.get(phrase, [])
        others = [n for n in muse_eye.CATALOG if n != phrase and n not in names]
        answer = None
        for i, f in enumerate(found[:3]):
            _, answer = muse_eye.confirm(rgb, f["box"], phrase)
            if any(n in answer for n in names) or answer in ("unavailable", ""):
                found.insert(0, found.pop(i))
                return found, answer
            if not any(o == answer or o in answer for o in others):
                return found, f"{answer} (unsure: kept the SAM box)"
        return [], f"{answer} (not a {phrase})"

    def look_in_basket(self, phrase):
        """The basket camera looks down into the robot's own basket; SAM 3.1 finds the item."""
        from perception import sam_find
        rgb, depth = self.camera("basket")
        self.last_basket_rgb = rgb
        for ph in ALIASES.get(phrase, [phrase]):
            boxes = sam_find(rgb, ph)
            if boxes:
                b = max(boxes, key=lambda b: b["conf"] or 0)
                f = {"pos": self._surface_point(b["box"], b["center"], depth),
                     "grasp": self._grasp_point(b["box"], b["center"], depth), "box": b["box"], "conf": b["conf"]}
                try:
                    import muse_eye
                    muse_eye.log("basket", phrase, rgb, [b], xyz=f["grasp"])
                except Exception:
                    pass
                return f
        try:
            import muse_eye
            muse_eye.log("basket", phrase, rgb, [])
        except Exception:
            pass
        return None

    def pick_from_back(self, xyz):
        """Turn the arm around (over its left shoulder) and pick the item out of the basket."""
        if self.held:
            return False, "already holding " + self.held[0]
        pre = list(HOME)
        pre[0] = 2.6
        self.glide(pre, 200)
        return self.pick_at(xyz)

    def pick_seen(self, phrase):
        found = self.perceive(phrase)
        self._pcache = None                    # after a grasp attempt, always look again
        if not found:
            return False, f"could not see '{phrase}'"
        ok, msg = self.pick_at(found[0]["grasp"])
        if not ok and msg.startswith("the fingers could not close"):
            # e.g. a flat box wider than the open gripper: look again (it may have moved) and grip it with the
            # hand turned 90 degrees, across its narrow side
            self.grip_yaw = math.pi / 2
            found = self.perceive(phrase)
            self._pcache = None
            if found:
                ok, msg = self.pick_at(found[0]["grasp"])
            self.grip_yaw = 0.0
        return ok, msg

    def place_on(self, phrase):
        found = self.perceive(phrase)
        if not found:
            return False, f"could not see '{phrase}'"
        x, y, ztop = found[0]["pos"]
        return self.place_at([x, y], ztop)

    def close(self):
        p.disconnect()
