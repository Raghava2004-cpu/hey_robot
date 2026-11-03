DRONE_METHODS = {"position", "ground_robot_position", "fly_to", "find_pad", "wait",
                 "set_wind_correction", "land_at"}
MAX_ACTIONS = 40

DRONE_DOC = """
robot.position() -> [x, y, z] of the drone (GPS)
robot.ground_robot_position() -> [x, y] of the ground robot's center (rough, from its odometry). The landing pad is about 0.38 m BEHIND the robot's center.
robot.fly_to(x, y, z) -> [x, y, z] where the drone is after flying toward the target for EXACTLY 3.0 seconds (z must be 0.5-2.0 m)
robot.find_pad() -> [x, y] of the orange landing pad seen by the downward camera (SAM 3.1), or None if not in view (from 1.3 m the camera sees about 0.5 m around the drone). Takes no time.
robot.wait(seconds) -> True   (hover in place while time passes; max 5 seconds)
robot.set_wind_correction(ax, ay) -> True   (extra acceleration in m/s^2 the autopilot adds, to cancel wind; limit +-3)
robot.land_at(x, y) -> True if it landed on the pad. The drone descends and reaches PAD HEIGHT EXACTLY 2.0 seconds after land_at starts, aiming at (x, y) the whole time. It lands only if the pad is under it at that moment (the pad is 0.45 m wide).
AUTOPILOT LAW (exact): acceleration = 6*(target - position) + 4*(target_velocity - velocity) + gravity + wind_correction + UNKNOWN_WIND.
So in a steady hover: position - target = (wind_correction + UNKNOWN_WIND) / 6.
The pad is on a ground robot that may be MOVING at constant velocity. Time passes ONLY during fly_to, wait and land_at; all other calls take no time.
Math helpers: math.sqrt, math.sin, math.cos, math.atan2, math.hypot, math.pi
"""


class DroneAPI:
    def __init__(self, env):
        self.env = env
        self.d = env.drone
        self.log = []

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")

    def position(self):
        return self.d.position()

    def ground_robot_position(self):
        x, y, _ = self.env.pose()
        return [round(x, 2), round(y, 2)]

    def fly_to(self, x, y, z):
        x, y, z = float(x), float(y), float(z)
        if not 0.5 <= z <= 2.0:
            self._record("fly_to", [x, y, z], "REFUSED: height must be 0.5-2.0 m")
            return self.d.position()
        pos = self.d.fly_to(x, y, z, 3.0)
        self._record("fly_to", f"target [{x:.2f}, {y:.2f}, {z:.2f}]", f"after 3 s the drone is at {pos}")
        return pos

    def find_pad(self):
        r = self.d.find("landing pad")
        xy = [r[0], r[1]] if r else None
        self._record("find_pad", "", xy if xy else "pad not in view")
        return xy

    def wait(self, seconds):
        s = min(max(float(seconds), 0.0), 5.0)
        self.env.step(int(s * 240))
        self._record("wait", s, f"drone at {self.d.position()}")
        return True

    def set_wind_correction(self, ax, ay):
        ax = max(-3.0, min(3.0, float(ax)))
        ay = max(-3.0, min(3.0, float(ay)))
        self.d.set_feedforward(ax, ay)
        self._record("set_wind_correction", [round(ax, 3), round(ay, 3)], "ok")
        return True

    def land_at(self, x, y):
        ok = self.d.land_at(float(x), float(y))
        td = self.d.position()
        self._record("land_at", f"[{float(x):.2f}, {float(y):.2f}]",
                     ("LANDED on the pad" if ok else "MISSED the pad") + f"; touched down at {td[:2]}")
        return ok