import math
import os

import numpy as np
import pybullet as p

DRONE_URDF = os.path.join("third_party", "gym-pybullet-drones", "gym_pybullet_drones",
                          "assets", "cf2x.urdf")
RL_POLICY = "rl_landing.npz"
MAX_PAYLOAD = 0.10        # kg: the vacuum gripper lifts light items only
KP, KD = 6.0, 4.0
W, H = 640, 480
ALIASES = {"landing pad": ["orange square", "orange pad", "orange landing pad", "orange box"]}


class Drone:
    """Simplified quadrotor: thrust from a PD autopilot, plus hidden wind and gusts."""

    def __init__(self, env, start=(-1.0, 1.0, 1.2), wind=(0.0, 0.0), scale=4):
        self.env = env
        self.body = p.loadURDF(DRONE_URDF, list(start), globalScaling=scale)
        p.changeDynamics(self.body, -1, angularDamping=0.9)
        links = range(-1, p.getNumJoints(self.body))
        self.mass = sum(p.getDynamicsInfo(self.body, j)[0] for j in links)
        self.bottom = start[2] - min(p.getAABB(self.body, j)[0][2] for j in links)
        self.wind = np.array([wind[0], wind[1], 0.0])   # HIDDEN from the AI (m/s^2)
        self.gust = np.zeros(3)                          # HIDDEN random gusts (m/s^2)
        self.gust_sigma, self.gust_tau = 0.0, 0.6
        self.rng = np.random.default_rng(0)
        self.ff = np.zeros(3)
        self.target = np.array(start, dtype=float)
        self.target_vel = np.zeros(3)
        self.rl_acc = None        # set only while the RL reflex is flying
        self.cargo = None         # (item name, constraint) while the gripper holds something
        self.payload = 0.0        # kg carried (the autopilot adds it to the thrust)
        self.detector = "sam"     # "sam" = SAM 3.1 (default); "mask" = simulator mask (fast experiments)
        self.motors_on = True
        self.latched = False
        self.touchdown_miss = None
        self.touchdown_xy = None
        self.last_rgb = None
        self.last_box = None
        self.track_log = []
        self.track_velocity = None
        self.sam_calls = 0
        self._seg = None
        self._frame = 0
        env.hooks.append(self._control)
        env.objects["drone"] = self.body

    # ---------- weather ----------
    def set_gusts(self, sigma, tau=0.6, seed=0):
        """Random gusts (Ornstein-Uhlenbeck), the same model the RL policy trained with."""
        self.gust_sigma, self.gust_tau = float(sigma), float(tau)
        self.rng = np.random.default_rng(seed)
        self.gust = np.zeros(3)
        if sigma > 0:
            self.gust[:2] = self.rng.normal(size=2) * sigma

    # ---------- autopilot (runs every physics step) ----------
    def _control(self):
        if self.gust_sigma > 0:
            dt = 1 / 240
            self.gust[:2] += (-self.gust[:2] / self.gust_tau * dt + self.gust_sigma
                              * math.sqrt(2 * dt / self.gust_tau) * self.rng.normal(size=2))
        if not self.motors_on:
            return
        pos, _ = p.getBasePositionAndOrientation(self.body)
        vel, _ = p.getBaseVelocity(self.body)
        pos, vel = np.array(pos), np.array(vel)
        acc = (KP * (self.target - pos) + KD * (self.target_vel - vel)
               + np.array([0, 0, 9.81]) + self.ff)
        if self.rl_acc is not None:          # the RL reflex commands the horizontal acceleration
            acc[0], acc[1] = self.rl_acc[0] + self.ff[0], self.rl_acc[1] + self.ff[1]
        force = (self.mass + self.payload) * (acc + self.wind + self.gust)
        p.applyExternalForce(self.body, -1, force.tolist(), pos.tolist(), p.WORLD_FRAME)

    # ---------- basic commands ----------
    def position(self):
        return [round(v, 3) for v in p.getBasePositionAndOrientation(self.body)[0]]

    def fly_to(self, x, y, z, seconds=3.0):
        self.motors_on = True
        self.target = np.array([x, y, z], dtype=float)
        self.target_vel = np.zeros(3)
        self.env.step(int(seconds * 240))
        return self.position()

    def set_feedforward(self, ax, ay):
        self.ff = np.array([ax, ay, 0.0])

    # ---------- cameras + SAM 3.1 ----------
    def camera_down(self):
        x, y, z = p.getBasePositionAndOrientation(self.body)[0]
        self._view = p.computeViewMatrix([x, y, z - 0.2], [x, y, 0.0], [1, 0, 0])
        self._proj = p.computeProjectionMatrixFOV(70, W / H, 0.05, 4.0)
        _, _, rgb, depth, seg = p.getCameraImage(W, H, self._view, self._proj,
                                                 renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        self._seg = np.reshape(seg, (H, W))
        return rgb, np.reshape(depth, (H, W))

    def camera_look(self, target):
        """Angled camera from the drone toward a target point (sees the sides of objects)."""
        x, y, z = p.getBasePositionAndOrientation(self.body)[0]
        self._view = p.computeViewMatrix([x, y, z - 0.2], list(target), [0, 0, 1])
        self._proj = p.computeProjectionMatrixFOV(60, W / H, 0.05, 5.0)
        _, _, rgb, depth, _ = p.getCameraImage(W, H, self._view, self._proj,
                                               renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        self.last_rgb = rgb
        return rgb, np.reshape(depth, (H, W))

    def _pixel_to_world(self, u, v, depth):
        V = np.array(self._view).reshape(4, 4, order="F")
        P = np.array(self._proj).reshape(4, 4, order="F")
        d = depth[int(v), int(u)]
        ndc = np.array([2 * u / W - 1, 1 - 2 * v / H, 2 * d - 1, 1.0])
        w = np.linalg.inv(P @ V) @ ndc
        return (w[:3] / w[3]).tolist()

    def _mask_boxes(self, phrase):
        """Fast stand-in for SAM in big experiments: the simulator's own object mask."""
        body = self.env.objects.get(phrase)
        if body is None or self._seg is None:
            return []
        ys, xs = np.nonzero(self._seg == body)
        if len(xs) < 15:
            return []
        k = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        return [{"box": [xs.min(), ys.min(), xs.max(), ys.max()], "center": [xs[k], ys[k]], "conf": 1.0}]

    def find(self, phrase):
        rgb, depth = self.camera_down()
        self.last_rgb = rgb
        self.last_box = None
        if self.detector == "mask":
            names = [phrase]
        else:
            from mobile_env import ALIASES as ITEM_ALIASES
            from perception import sam_find
            names = ALIASES.get(phrase) or ITEM_ALIASES.get(phrase, [phrase])
        for ph in names:
            if self.detector == "mask":
                boxes = self._mask_boxes(ph)
            else:
                self.sam_calls += 1
                boxes = sam_find(rgb, ph)
            if boxes:
                b = max(boxes, key=lambda b: b["conf"] or 0)
                self.last_box = b["box"]
                xyz = [round(c, 3) for c in self._pixel_to_world(*b["center"], depth)]
                self._log(ph, rgb, [b], xyz, phrase)
                return xyz
        self._log(names[0], rgb, [], None, phrase)
        return None

    def _log(self, ph, rgb, boxes, xyz, phrase):
        """Every drone detection goes to the vision log (pictures: items always, the pad every 4th look)."""
        if self.detector == "mask":
            return
        try:
            import muse_eye
            muse_eye.log("drone", ph, rgb, boxes, xyz=xyz,
                         save=(phrase != "landing pad" or self.sam_calls % 4 == 0))
        except Exception:
            pass

    # ---------- landing helpers ----------
    def _latch(self):
        """Centering latch (like a real drone dock): pull the drone to the pad center and lock it."""
        env = self.env
        rel = [env.pad_offset[0], env.pad_offset[1], env.pad_offset[2] + 0.006 + self.bottom]
        c = p.createConstraint(env.husky, -1, self.body, -1, p.JOINT_FIXED,
                               [0, 0, 0], rel, [0, 0, 0])
        p.changeConstraint(c, maxForce=2000)
        for link in range(-1, p.getNumJoints(self.body)):
            p.setCollisionFilterPair(self.body, env.objects["landing pad"], link, -1, 0)
            for hl in range(-1, p.getNumJoints(env.husky)):
                p.setCollisionFilterPair(self.body, env.husky, link, hl, 0)
            for pl in range(-1, p.getNumJoints(env.panda)):
                p.setCollisionFilterPair(self.body, env.panda, link, pl, 0)
        self.latched = True
        self.motors_on = False
        env.set_wheels(0, 0)               # landing confirmed -> the ground robot stops
        env.say("LANDED!")
        env.follow_camera = False

    def _start_landing(self):
        cine = self.env.gui
        if cine:                           # cinematic close-up in 3x slow motion
            self.env.follow_camera = False
            self.env.slowmo = 3.0
        self.motors_on = True
        self.rl_acc = None
        self._frame = 0
        self.track_log = []
        self.track_velocity = None
        return cine

    def _cine(self):
        if not self.env.gui:
            return
        self._frame += 1
        if self._frame % 4 == 0:
            d = p.getBasePositionAndOrientation(self.body)[0]
            p.resetDebugVisualizerCamera(1.6, 50, -18, [d[0], d[1], max(d[2] - 0.3, 0.4)])

    def _finish_landing(self, cine):
        self.rl_acc = None
        d = p.getBasePositionAndOrientation(self.body)[0]
        pad = p.getBasePositionAndOrientation(self.env.objects["landing pad"])[0]
        self.touchdown_xy = [round(d[0], 3), round(d[1], 3)]
        self.touchdown_miss = math.hypot(d[0] - pad[0], d[1] - pad[1])
        landed = self.touchdown_miss <= self.env.pad_half and d[2] > self.env.pad_top - 0.05
        if landed:
            self._latch()
        else:
            self.motors_on = False
            self.target_vel = np.zeros(3)
            self.env.say("MISSED the pad")
        if cine:
            self.env.slowmo = 1.0
            self.env.step(360)             # hold the final shot for 1.5 s
            if landed:
                d = p.getBasePositionAndOrientation(self.body)[0]
                p.resetDebugVisualizerCamera(1.8, 50, -20, [d[0], d[1], 0.5])
        if not landed:
            self.env.step(120)
        return landed

    # ---------- open-loop landing: aim once, then descend blind ----------
    def land_at(self, x, y, seconds=2.0):
        """Reach PAD HEIGHT exactly `seconds` after starting, aiming at a fixed (x, y)."""
        cine = self._start_landing()
        z0 = p.getBasePositionAndOrientation(self.body)[0][2]
        z_contact = self.env.pad_top + self.bottom - 0.03
        vz = (z_contact - z0) / seconds
        for i in range(int(seconds * 240) + 240):            # +1 s in case it misses
            zt = max(z0 + vz * (i + 1) / 240, 0.0)
            self.target = np.array([x, y, zt])
            self.target_vel = np.array([0.0, 0.0, vz])
            self.env.step(1)
            self._cine()
            if p.getContactPoints(self.body):
                break
        return self._finish_landing(cine)

    # ---------- closed-loop landing: keep re-finding the pad (Keipour et al. idea) ----------
    def _pad_xy(self):
        """(xy, clean): clean = the whole pad is inside the image, so its center is unbiased."""
        z = p.getBasePositionAndOrientation(self.body)[0][2]
        if z - 0.2 - self.env.pad_top < 0.45:              # too close: the pad no longer fits
            return None, False
        try:
            r = self.find("landing pad")
        except Exception:
            return None, False
        if r is None or self.last_box is None:
            return None, False
        x0, y0, x1, y1 = [float(v) for v in self.last_box]
        clean = (x0 > 4 and y0 > 4 and x1 < W - 4 and y1 < H - 4
                 and x1 - x0 < 0.9 * W and y1 - y0 < 0.9 * H)
        return [r[0], r[1]], clean

    @staticmethod
    def _fit_velocity(hist, prior):
        """Least-squares pad velocity from the last clean sightings (needs >= 3 over >= 0.5 s)."""
        pts = hist[-6:]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.5:
            return prior
        a = np.array(pts)
        t = a[:, 0] - a[:, 0].mean()
        den = (t * t).sum()
        return np.array([(t * (a[:, 1] - a[:, 1].mean())).sum() / den,
                         (t * (a[:, 2] - a[:, 2].mean())).sum() / den])

    def _ready(self, hist, aim):
        d = p.getBasePositionAndOrientation(self.body)[0]
        return (len(hist) >= 3 and hist[-1][0] - hist[0][0] >= 0.5
                and math.hypot(d[0] - aim[0], d[1] - aim[1]) < 0.08)

    def land_tracking(self, vx=0.0, vy=0.0, track_s=1.0, every=0.25, seconds=2.0, max_track_s=3.0):
        """Catch up and follow the pad (re-finding it with SAM 3.1 every `every` s), then descend to
        pad height in exactly `seconds` s while still correcting the aim. (vx, vy) = starting guess."""
        cine = self._start_landing()
        v = np.array([float(vx), float(vy)])
        k = max(1, int(every * 240))
        hist = []
        d = p.getBasePositionAndOrientation(self.body)[0]
        est, t_est, z_hover = np.array(d[:2]), 0.0, d[2]
        r, clean = self._pad_xy()
        if r is not None:
            est = np.array(r)
            if clean:
                hist.append((0.0, r[0], r[1]))

        def sight(t_now):
            nonlocal est, t_est, v
            r, clean = self._pad_xy()
            if r is None:
                return
            est, t_est = np.array(r), t_now
            if clean:
                hist.append((t_now, r[0], r[1]))
                v = self._fit_velocity(hist, v)

        # phase 1: catch up and follow at the current height until the velocity is measured
        i, t = 0, 0.0
        while i < int(max_track_s * 240):
            i += 1
            t = i / 240
            if i % k == 0:
                sight(t)
            aim = est + v * (t - t_est)
            self.target = np.array([aim[0], aim[1], z_hover])
            self.target_vel = np.array([v[0], v[1], 0.0])
            self.env.step(1)
            self._cine()
            if i >= int(track_s * 240) and self._ready(hist, aim):
                break

        # phase 2: descend, still re-finding the pad while it fits in the view
        z0 = p.getBasePositionAndOrientation(self.body)[0][2]
        z_contact = self.env.pad_top + self.bottom - 0.03
        vz = (z_contact - z0) / seconds
        t1 = t
        for i in range(1, int(seconds * 240) + 241):
            t = t1 + i / 240
            if i % k == 0:
                sight(t)
            aim = est + v * (t - t_est)
            zt = max(z0 + vz * i / 240, 0.0)
            self.target = np.array([aim[0], aim[1], zt])
            self.target_vel = np.array([v[0], v[1], vz])
            self.env.step(1)
            self._cine()
            if p.getContactPoints(self.body):
                break
        self.track_log = [[round(a, 2), round(b, 3), round(c, 3)] for a, b, c in hist]
        self.track_velocity = [round(float(v[0]), 3), round(float(v[1]), 3)]
        return self._finish_landing(cine)

    # ---------- learned landing reflex: PPO policy trained on 4,096 GPU drones ----------
    def _policy(self):
        if not hasattr(self, "_pi"):
            w = np.load(RL_POLICY)
            self._pi = {k: w[k] for k in w.files}
        return self._pi

    def _act(self, obs):
        w = self._pi
        h = np.tanh(w["l1_w"] @ obs + w["l1_b"])
        h = np.tanh(w["l2_w"] @ h + w["l2_b"])
        return float(w["a_max"]) * np.tanh(w["l3_w"] @ h + w["l3_b"])

    def land_rl(self, vx=0.0, vy=0.0, every=0.25):
        """The RL reflex flies the last 3 s: follow the pad 1 s, then descend 2 s. It sees what the
        drone sees: a pad sighting every 0.25 s, its own velocity, and the gusts it feels."""
        w = self._policy()
        dt, a_max = float(w["dt"]), float(w["a_max"])
        t_track, t_desc = float(w["t_track"]), float(w["t_desc"])
        steps = int(round((t_track + t_desc) / dt))
        sub = int(round(dt * 240))                         # 6 physics steps per decision (40 Hz)
        sight_every = int(round(every / dt))
        cine = self._start_landing()
        pos, _ = p.getBasePositionAndOrientation(self.body)
        z_contact = self.env.pad_top + self.bottom - 0.03
        h0 = pos[2] - z_contact
        vest = np.array([float(vx), float(vy)])
        est, t_est, hist = np.array(pos[:2]), 0.0, []
        r, clean = self._pad_xy()
        if r is not None:
            est = np.array(r)
            if clean:
                hist.append((0.0, r[0], r[1]))
        dist, a_prev = np.zeros(2), np.zeros(2)
        v_prev = np.array(p.getBaseVelocity(self.body)[0][:2])

        def height(t):
            return h0 if t <= t_track else h0 * max(0.0, 1.0 - (t - t_track) / t_desc)

        touched = False
        for k in range(steps + 40):                       # +1 s in case it has not touched yet
            t = min(k, steps) * dt
            pos = np.array(p.getBasePositionAndOrientation(self.body)[0][:2])
            vel = np.array(p.getBaseVelocity(self.body)[0][:2])
            est_now = est + vest * (k * dt - t_est)
            obs = np.concatenate([(est_now - pos) / 0.3, (vest - vel) / 0.3, dist / 1.5,
                                  [height(t) / 0.9, 1 - t / (t_track + t_desc)], a_prev])
            a = self._act(obs)
            self.rl_acc = a
            t_next = min(k + 1, steps) * dt
            vz = -h0 / t_desc if t_track < t_next <= t_track + t_desc else 0.0
            self.target = np.array([0.0, 0.0, z_contact + height(t_next)])
            self.target_vel = np.array([0.0, 0.0, vz])
            for _ in range(sub):
                self.env.step(1)
                self._cine()
                if p.getContactPoints(self.body):
                    touched = True
                    break
            if touched:
                break
            vel = np.array(p.getBaseVelocity(self.body)[0][:2])
            dist = 0.7 * dist + 0.3 * ((vel - v_prev) / dt - a)   # gusts felt, like an accelerometer
            v_prev, a_prev = vel, a / a_max
            if (k + 1) % sight_every == 0:
                r, clean = self._pad_xy()
                if r is not None:
                    est, t_est = np.array(r), (k + 1) * dt
                    if clean:
                        hist.append(((k + 1) * dt, r[0], r[1]))
                        vest = self._fit_velocity(hist, vest)
        self.track_log = [[round(a, 2), round(b, 3), round(c, 3)] for a, b, c in hist]
        self.track_velocity = [round(float(vest[0]), 3), round(float(vest[1]), 3)]
        return self._finish_landing(cine)

    # ---------- vacuum gripper: pick up light items where the arm cannot reach ----------
    def align_over(self, phrase, z, tries=6, tol=0.015):
        """Visual servoing: look down with SAM 3.1 and shift until the drone is right above the thing.
        Works in unknown wind because it corrects the measured offset, not the commanded one."""
        target = np.array(self.target[:2], dtype=float)
        if abs(self.position()[2] - z) > 0.03:                    # go to the viewing height first
            self.target = np.array([target[0], target[1], z])
            self.target_vel = np.zeros(3)
            self.env.step(360)
        seen = None
        for _ in range(tries):
            r = self.find(phrase)
            if r is None:
                return None
            seen = r
            pos = np.array(self.position()[:2])
            err = np.array(r[:2]) - pos
            if np.linalg.norm(err) < tol:
                break
            target = target + err
            self.target = np.array([target[0], target[1], z])
            self.target_vel = np.zeros(3)
            self.env.step(300)
        return seen

    def grab(self, phrase):
        """Align above the item, come down until the gripper touches it, switch the suction on, climb."""
        if self.cargo:
            return False, f"already carrying the {self.cargo[0]}"
        body = self.env.objects.get(phrase)
        start_z = self.position()[2]
        seen = self.align_over(phrase, start_z)
        if seen is None or body is None:
            return False, f"could not see '{phrase}' below the drone"
        mass = p.getDynamicsInfo(body, -1)[0]
        if mass > MAX_PAYLOAD:
            return False, f"the {phrase} is too heavy for the drone ({mass * 1000:.0f} g; limit {MAX_PAYLOAD * 1000:.0f} g)"
        top = seen[2]
        seen = self.align_over(phrase, top + 0.40) or seen          # closer look, finer aim
        z0, z1 = self.position()[2], top + self.bottom + 0.004
        hold = self.target[:2].copy()
        for i in range(1, 721):                                     # gentle descent (up to 3 s)
            zt = z0 + (z1 - z0) * min(1.0, i / 360)
            self.target = np.array([hold[0], hold[1], zt])
            self.target_vel = np.zeros(3)
            self.env.step(1)
            if p.getContactPoints(self.body, body) or self.position()[2] <= z1 + 0.006:
                break
        d = np.array(self.position())
        o = np.array(p.getBasePositionAndOrientation(body)[0])
        if np.linalg.norm(d[:2] - o[:2]) > 0.06 or d[2] - self.bottom - o[2] > 0.08:
            self.fly_to(hold[0], hold[1], start_z, 2.0)
            return False, "missed: the gripper did not reach the item"
        dp, do = p.getBasePositionAndOrientation(self.body)
        ip, io = p.invertTransform(dp, do)
        rp, ro = p.multiplyTransforms(ip, io, *p.getBasePositionAndOrientation(body))
        c = p.createConstraint(self.body, -1, body, -1, p.JOINT_FIXED, [0, 0, 0], rp, [0, 0, 0], ro)
        p.changeConstraint(c, maxForce=50)
        for link in range(-1, p.getNumJoints(self.body)):
            p.setCollisionFilterPair(self.body, body, link, -1, 0)
        self.cargo, self.payload = (phrase, c), mass
        self.env.say(f"drone grabbed the {phrase}")
        self.fly_to(hold[0], hold[1], start_z, 2.0)
        return True, f"grabbed the {phrase} ({mass * 1000:.0f} g)"

    def drop_on_moving(self, vx=0.0, vy=0.0, track_s=1.0, desc_s=2.0, every=0.25, above=0.14, max_err=0.10):
        """Hand-off on the move. The robot keeps driving; the drone
        1) follows the robot's orange pad with SAM 3.1 (a sighting every 0.25 s) and fits its velocity,
        2) comes down while predicting where the pad will be (sightings continue while the pad fits),
        3) lets go of the item just above the basket, matching the robot's speed, then climbs away.
        vx, vy = the robot's speed from its wheel odometry (a first guess; SAM sightings refine it).
        Returns (ok, message, info)."""
        if not self.cargo:
            return False, "the drone is not carrying anything", {}
        env = self.env
        name, c = self.cargo
        body = env.objects[name]
        sub = int(round(every * 240))
        z_high = env.pad_top + 0.95                         # high enough that the whole pad stays in view
        hist, est, t_est, t = [], None, 0.0, 0.0
        vest = np.array([float(vx), float(vy)])             # the robot's own speed report (odometry) as a start
        # wind drift: in the hover just before, the drone sat (target - position) away from its target
        bias = np.clip(np.array(self.target[:2], dtype=float) - np.array(self.position()[:2]), -0.3, 0.3)

        def sight(now):
            nonlocal est, t_est, vest
            r, clean = self._pad_xy()
            if r is not None:
                est, t_est = np.array(r), now
                if clean:
                    hist.append((now, r[0], r[1]))
                    vest = self._fit_velocity(hist, vest)

        def follow(now, z, vz=0.0):
            nonlocal bias
            pred = est + vest * (now - t_est)
            pos = np.array(self.position()[:2])
            e = pred - pos
            if np.linalg.norm(e) < 0.10:                    # slow integral (only near the goal: no wind-up)
                bias = np.clip(bias + 0.003 * e, -0.3, 0.3)
            self.target = np.array([pred[0] + bias[0], pred[1] + bias[1], z])
            self.target_vel = np.array([vest[0], vest[1], vz])
            return pred, pos

        # 1) track from above
        for k in range(int(round(track_s / every))):
            sight(t)
            if est is not None:
                for _ in range(sub):
                    follow(t, z_high)
                    env.step(1)
                    t += 1 / 240
            else:
                env.step(sub)
                t += every
        if est is None:
            return False, "could not see the robot's pad", {}
        # 2) come down while predicting the pad
        hang = self.position()[2] - p.getAABB(body)[0][2]
        z_drop = env.pad_top + above + hang
        n = int(round(desc_s * 240))
        for i in range(n):
            a = min(1.0, i / (0.8 * n))
            z = z_high + (z_drop - z_high) * a
            if i % sub == 0 and z - 0.2 - env.pad_top >= 0.45:
                sight(t)
            pred, pos = follow(t, z, (z_drop - z_high) / (0.8 * desc_s) if a < 1 else 0.0)
            env.step(1)
            t += 1 / 240
        # 3) let go only when the item is over the basket (up to 1 s more to line up)
        err = 9.9
        for i in range(240):
            pred, pos = follow(t, z_drop)
            item = np.array(p.getBasePositionAndOrientation(body)[0][:2])
            err = float(np.linalg.norm(item - pred))
            if err < max_err:
                break
            env.step(1)
            t += 1 / 240
        info = {"sightings": len(hist), "velocity": [round(float(v), 3) for v in vest],
                "aim_error_cm": round(err * 100, 1)}
        if err >= max_err:
            self.target = np.array([pos[0], pos[1], z_high])
            self.target_vel = np.zeros(3)
            env.step(240)
            return False, f"could not line up over the basket (off by {err * 100:.0f} cm); still holding the {name}", info
        p.removeConstraint(c)
        self.cargo, self.payload = None, 0.0
        for _ in range(72):                                 # keep pace for 0.3 s while the item falls
            follow(t, z_drop)
            env.step(1)
            t += 1 / 240
        x, y, _ = self.position()
        self.target = np.array([x, y, max(z_high, 1.3)])   # climb straight up, away from the robot
        self.target_vel = np.zeros(3)
        env.step(240)
        for link in range(-1, p.getNumJoints(self.body)):
            p.setCollisionFilterPair(self.body, body, link, -1, 1)
        env.say(f"drone dropped the {name} into the moving basket")
        return True, f"dropped the {name} over the moving robot (aim {err * 100:.1f} cm)", info

    def drop_into(self, phrase, above=0.22):
        """Align above the container with SAM 3.1, come down close, switch the suction off."""
        if not self.cargo:
            return False, "the drone is not carrying anything"
        z_start = self.position()[2]
        seen = self.align_over(phrase, z_start)
        if seen is None:
            return False, f"could not see '{phrase}' below the drone"
        name, c = self.cargo
        body = self.env.objects[name]
        hang = self.position()[2] - p.getAABB(body)[0][2]            # how far the item hangs below
        z_rel = seen[2] + above + hang
        seen = self.align_over(phrase, z_rel) or seen
        self.fly_to(self.target[0], self.target[1], z_rel, 1.5)
        p.removeConstraint(c)
        self.cargo, self.payload = None, 0.0
        self.env.step(240)
        self.env.say(f"drone dropped the {name}")
        self.fly_to(self.target[0], self.target[1], z_start, 2.0)
        for link in range(-1, p.getNumJoints(self.body)):
            p.setCollisionFilterPair(self.body, body, link, -1, 1)
        return True, f"dropped the {name} into the {phrase}"
