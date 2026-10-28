import time
import numpy as np
import pybullet as p
import pybullet_data

EE_LINK = 11
DOWN = p.getQuaternionFromEuler([3.14159, 0, 0])
HOME = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]
TABLE_Z = 0.625
W, H = 640, 480


class RobotEnv:
    def __init__(self, gui=True):
        self.gui = gui
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, 0])
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, TABLE_Z], useFixedBase=True)
        for j, q in enumerate(HOME):
            p.resetJointState(self.robot, j, q)
        self.objects = {}
        self.held = None  # (name, constraint_id)
        self.objects["tray"] = p.loadURDF("tray/traybox.urdf", [0.6, -0.25, TABLE_Z], globalScaling=0.6)

    # ---------- helpers ----------
    def add_cube(self, name, xy, rgba):
        body = p.loadURDF("cube_small.urdf", [xy[0], xy[1], TABLE_Z + 0.03])
        p.changeVisualShape(body, -1, rgbaColor=rgba)
        self.objects[name] = body

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)

    def get_pos(self, name):
        return list(p.getBasePositionAndOrientation(self.objects[name])[0])

    def hand_pos(self):
        return list(p.getLinkState(self.robot, EE_LINK)[4])

    # ---------- basic actions ----------
    def move_to(self, target, steps=200):
        for _ in range(steps):
            joints = p.calculateInverseKinematics(self.robot, EE_LINK, target, DOWN, maxNumIterations=50)
            for j in range(7):
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joints[j],
                                        force=240, maxVelocity=1.5)
            self.step()
        err = np.linalg.norm(np.array(self.hand_pos()) - np.array(target))
        return err < 0.02

    def gripper(self, open_):
        width = 0.04 if open_ else 0.0
        for j in (9, 10):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, width, force=20)
        self.step(60)

    def pick(self, name):
        if self.held:
            return False, "already holding " + self.held[0]
        x, y, z = self.get_pos(name)
        self.gripper(True)
        self.move_to([x, y, z + 0.15])
        self.move_to([x, y, z + 0.01])
        # sticky gripper: attach only if the hand is really close
        dist = np.linalg.norm(np.array(self.hand_pos()) - np.array(self.get_pos(name)))
        if dist > 0.04:
            self.move_to([x, y, z + 0.15])
            return False, f"hand too far from {name} ({dist:.3f} m)"
        self.gripper(False)
        hand = p.getLinkState(self.robot, EE_LINK)
        inv_pos, inv_orn = p.invertTransform(hand[4], hand[5])
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.objects[name])
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
        c = p.createConstraint(self.robot, EE_LINK, self.objects[name], -1, p.JOINT_FIXED,
                               [0, 0, 0], rel_pos, [0, 0, 0], rel_orn)
        self.held = (name, c)
        self.move_to([x, y, z + 0.15])
        return True, "picked " + name

    def place(self, xy, z=None):
        if not self.held:
            return False, "not holding anything"
        name = self.held[0]
        drop_z = (z if z is not None else TABLE_Z + 0.03) + 0.03
        self.move_to([xy[0], xy[1], drop_z + 0.12])
        self.move_to([xy[0], xy[1], drop_z])
        p.removeConstraint(self.held[1])
        self.held = None
        self.gripper(True)
        self.step(60)
        self.move_to([xy[0], xy[1], drop_z + 0.12])
        return True, "placed " + name
    
        # ---------- camera ----------
    def camera(self):
        self._view = p.computeViewMatrix([0.55, 0, 1.45], [0.55, 0, TABLE_Z], [-1, 0, 0])
        self._proj = p.computeProjectionMatrixFOV(60, W / H, 0.1, 3.0)
        _, _, rgb, depth, _ = p.getCameraImage(W, H, self._view, self._proj,
                                               renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth, (H, W))
        return rgb, depth

    def _matrix(self):
        V = np.array(self._view).reshape(4, 4, order="F")
        P = np.array(self._proj).reshape(4, 4, order="F")
        return P @ V

    def pixel_to_world(self, u, v, depth):
        d = depth[int(v), int(u)]
        ndc = np.array([2 * u / W - 1, 1 - 2 * v / H, 2 * d - 1, 1.0])
        w = np.linalg.inv(self._matrix()) @ ndc
        return (w[:3] / w[3]).tolist()

    def world_to_pixel(self, xyz):
        clip = self._matrix() @ np.array([*xyz, 1.0])
        ndc = clip[:3] / clip[3]
        return (ndc[0] + 1) / 2 * W, (1 - ndc[1]) / 2 * H