import math
import os

import pybullet as p

from mobile_env import ALIASES, MobileEnv
from team_skills import TeamAPI

YCB = os.path.join("third_party", "pybullet-object-models", "pybullet_object_models", "ycb_objects")
BALLS = ["tennis ball", "tennis ball 2", "tennis ball 3"]
for _b in BALLS:
    ALIASES[_b] = ["tennis ball", "yellow ball", "ball"]
ALIASES["bin"] = ["blue bin", "blue box", "blue basket"]

BIN_HALF, BIN_RIM = 0.2, 0.18
RELEASE = (0.45, 0.0, 0.95)       # where the ball leaves the hand, in the robot's frame (x forward)
WINDUP = (0.0, 0.0, 1.0)

THROW_METHODS = {"places", "go_to", "pick", "put_aside", "is_holding", "ground_robot_position",
                 "see", "release_point", "predict_distance", "throw", "throws"}
THROW_DOC = """
GROUND ROBOT (wheeled robot with an arm; holds ONE ball at a time):
robot.places() -> list of place names: ["ball rack", "throw line"]
robot.go_to(place) -> True/False   ("ball rack": face the table with the balls; "throw line": stand facing the bin)
robot.pick(item) -> True/False   (pick the named item from the table in front, e.g. "tennis ball")
robot.is_holding() -> True/False
robot.ground_robot_position() -> [x, y] of the robot's center
robot.see(thing) -> [x, y, z] of the thing seen by the robot's front camera (SAM 3.1), or None. Use "blue bin" for the bin.
THROWING:
robot.release_point() -> [x, y, z] where the ball leaves the hand when throwing from the current pose
robot.predict_distance(speed, angle_deg, target_z) -> horizontal distance (m) the ball WOULD fly from the release point
   until it comes down to height target_z, using simple projectile math WITHOUT air drag
robot.throw(speed, angle_deg) -> dict {"distance": horizontal distance flown from the release point until the ball
   first touched something, "landed_at": [x, y], "in_bin": True/False}. The robot must hold a ball; it throws
   straight ahead (the direction it faces) at `speed` m/s (1.0-6.0) and `angle_deg` above horizontal (10-70).
robot.throws() -> list of all throws so far (speed, angle, distance, in_bin)
Physics: gravity 9.81 m/s^2. The air slows the ball by an UNKNOWN amount, so real throws can land shorter than
predict_distance says. The bin is 0.40 m wide and 0.18 m tall; aim for its center at about target_z = 0.15.
Math helpers: math.sqrt, math.sin, math.cos, math.atan2, math.hypot, math.pi, math.radians
"""
THROW_TEXT = ("Throw a tennis ball into the blue bin. The tennis balls are on the ball rack table "
              "(there are 3). Pick one, go to the throw line, and throw it into the bin. The air slows the "
              "ball by an unknown amount, so plain projectile math may miss: learn from each throw.")


class ThrowAPI(TeamAPI):
    def __init__(self, env):
        super().__init__(env)
        self.throw_log = []

    def _look_ahead(self, thing):
        """Head camera looking far ahead (the normal front camera only sees the table in front)."""
        import numpy as np
        from perception import sam_find
        env = self.env
        x, y, yaw = env.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        env._view = p.computeViewMatrix([x + 0.3 * c, y + 0.3 * s, 1.3],
                                        [x + 2.3 * c, y + 2.3 * s, 0.0], [0, 0, 1])
        env._proj = p.computeProjectionMatrixFOV(60, 640 / 480, 0.1, 8.0)
        _, _, rgb, depth, _ = p.getCameraImage(640, 480, env._view, env._proj, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (480, 640, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth, (480, 640))
        for ph in ALIASES.get(thing, [thing]):
            boxes = sam_find(rgb, ph)
            if boxes:
                b = max(boxes, key=lambda b: b["conf"] or 0)
                if thing != "bin":
                    return env.pixel_to_world(*b["center"], depth)
                # an open bin: its center is halfway between the near edge (bottom of the box)
                # and the far rim (top of the box)
                x0, y0, x1, y1 = [float(v) for v in b["box"]]
                uc = (x0 + x1) / 2
                near = env.pixel_to_world(uc, min(y1 - 2, 479), depth)
                far = env.pixel_to_world(uc, max(y0 + 2, 0), depth)
                return [(near[0] + far[0]) / 2, (near[1] + far[1]) / 2, BIN_RIM]
        return None

    def see(self, thing):
        thing = str(thing)
        xyz = self._look_ahead("bin" if "bin" in thing else thing)
        if xyz is None:
            found = self.env.perceive(thing)
            xyz = found[0]["pos"] if found else None
        xyz = [round(c, 3) for c in xyz] if xyz else None
        self._record("see", thing, xyz if xyz else "not seen")
        return xyz

    def release_point(self):
        x, y, yaw = self.env.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        return [round(x + RELEASE[0] * c, 3), round(y + RELEASE[0] * s, 3), RELEASE[2]]

    def predict_distance(self, speed, angle_deg, target_z=0.15):
        v, a = float(speed), math.radians(float(angle_deg))
        vx, vz = v * math.cos(a), v * math.sin(a)
        dz = RELEASE[2] - float(target_z)
        disc = vz * vz + 2 * 9.81 * dz
        if disc < 0:
            return 0.0
        return round(vx * (vz + math.sqrt(disc)) / 9.81, 3)

    def throw(self, speed, angle_deg):
        env = self.env
        speed = min(max(float(speed), 1.0), 6.0)
        angle = min(max(float(angle_deg), 10.0), 70.0)
        if not env.held or not env.held[0].startswith("tennis ball"):
            self._record("throw", [speed, angle], "REFUSED: not holding a tennis ball")
            return {"distance": 0.0, "landed_at": None, "in_bin": False}
        name, c = env.held
        ball = env.objects[name]
        x, y, yaw = env.pose()
        cy, sy = math.cos(yaw), math.sin(yaw)

        def world(rel):
            return [x + rel[0] * cy - rel[1] * sy, y + rel[0] * sy + rel[1] * cy, rel[2]]

        env.say(f"throw: {speed:.2f} m/s at {angle:.0f} deg")
        env.move_arm_to(world(WINDUP), steps=120)                       # wind up
        start, end = world(WINDUP), world(RELEASE)
        for i in range(1, 25):                                          # fast forward swing
            f = i / 24
            tgt = [start[k] + (end[k] - start[k]) * f for k in range(3)]
            q = p.calculateInverseKinematics(env.panda, 11, tgt, env.down(),
                                             maxNumIterations=50)
            env.arm_q = list(q[:7])
            env.step(1)
        # release: the hand lets go at the release point with the commanded speed and angle
        p.removeConstraint(c)
        env.held = None
        env.finger = 0.04
        for link in range(-1, p.getNumJoints(env.panda)):
            p.setCollisionFilterPair(ball, env.panda, -1, link, 0)
        for link in range(-1, p.getNumJoints(env.husky)):
            p.setCollisionFilterPair(ball, env.husky, -1, link, 0)
        rp = self.release_point()
        a = math.radians(angle)
        v = [speed * math.cos(a) * cy, speed * math.cos(a) * sy, speed * math.sin(a)]
        p.resetBasePositionAndOrientation(ball, rp, [0, 0, 0, 1])
        p.resetBaseVelocity(ball, v, [0, 0, 0])
        env.flying.add(name)
        landed = None
        for _ in range(int(3.0 * 240)):
            env.step(1)
            if p.getContactPoints(bodyA=ball):
                landed = env.get_pos(name)
                break
        env.flying.discard(name)
        env.step(int(1.5 * 240))                                        # let it settle
        env.arm_home()
        if landed is None:
            landed = env.get_pos(name)
        dist = math.hypot(landed[0] - rp[0], landed[1] - rp[1])
        inside = ball_in_bin(env, name)
        res = {"distance": round(dist, 3), "landed_at": [round(landed[0], 3), round(landed[1], 3)],
               "in_bin": inside}
        self.throw_log.append({"speed": round(speed, 3), "angle": round(angle, 1), **res})
        self._record("throw", [round(speed, 3), round(angle, 1)], res)
        env.say("IN THE BIN!" if inside else f"missed: flew {dist:.2f} m")
        return res

    def throws(self):
        return [dict(t) for t in self.throw_log]


def ball_in_bin(env, name):
    x, y, z = env.get_pos(name)
    bx, by = env.bin_center
    return abs(x - bx) < BIN_HALF - 0.01 and abs(y - by) < BIN_HALF - 0.01 and z < BIN_RIM


def add_bin(env, center, half=BIN_HALF, rim=BIN_RIM, wall=0.012, rgba=(0.1, 0.3, 0.9, 1)):
    cx, cy = center
    parts = [((half, half, wall / 2), (cx, cy, wall / 2)),
             ((half, wall / 2, rim / 2), (cx, cy + half, rim / 2)),
             ((half, wall / 2, rim / 2), (cx, cy - half, rim / 2)),
             ((wall / 2, half, rim / 2), (cx + half, cy, rim / 2)),
             ((wall / 2, half, rim / 2), (cx - half, cy, rim / 2))]
    ids = []
    for half_ext, pos in parts:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=rgba)
        ids.append(p.createMultiBody(0, col, vis, pos))
    env.objects["bin"] = ids[0]
    env.static.add("bin")
    env.bin_center = (cx, cy)


def throw_scene(distance=2.2, drag=0.2):
    """distance = from the throw line (robot center) to the bin center. drag is HIDDEN from the AI."""
    def build(gui=False):
        env = MobileEnv(gui=gui)
        rack = env.add_station("ball rack", (0.0, -1.6), -math.pi / 2, rgba=(0.45, 0.35, 0.25, 1))
        line = (-0.5, 0.3)
        env.add_location("throw line", (line[0], line[1], 0.0), [])
        add_bin(env, (line[0] + distance, line[1]))
        for name, (bx, by) in zip(BALLS, rack["items"]):
            body = p.loadURDF(os.path.join(YCB, "YcbTennisBall", "model.urdf"), [bx, by, rack["top"] + 0.04])
            p.changeDynamics(body, -1, rollingFriction=0.02, spinningFriction=0.01, restitution=0.3)
            env.objects[name] = body
        line_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.5, 0.001], rgbaColor=[1, 1, 1, 1])
        p.createMultiBody(0, -1, line_v, [line[0] + 0.55, line[1], 0.002])     # painted throw line
        env.flying = set()
        env.drone = None
        env.station_centers = {"ball rack": (0.0, -1.6)}
        env.station_yaws = {"ball rack": -math.pi / 2}

        def air_drag():
            for n in env.flying:
                b = env.objects[n]
                v, _ = p.getBaseVelocity(b)
                sp = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                m = p.getDynamicsInfo(b, -1)[0]
                pos = p.getBasePositionAndOrientation(b)[0]
                p.applyExternalForce(b, -1, [-m * drag * sp * c for c in v], pos, p.WORLD_FRAME)
        env.hooks.append(air_drag)
        bx, by = env.bin_center
        env.views["bin"] = ([bx - 0.75, by - 0.45, 1.15], [bx, by, 0.05])   # close look into the bin
        env.main_view = "bin"
        if gui:
            p.resetDebugVisualizerCamera(3.6, 20, -35, [0.6, -0.2, 0.3])
        env.step(240)
        return env
    return build


def throw_check(env):
    inside = [n for n in BALLS if ball_in_bin(env, n)]
    return bool(inside), f"tennis balls in the bin: {inside or 'none'}"
