import time

import pybullet as p
import pybullet_data

HOME = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=50, cameraPitch=-30,
                             cameraTargetPosition=[1.2, 0, 0.3])

# --- Husky mobile base ---
husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
wheels = {}
for j in range(p.getNumJoints(husky)):
    name = p.getJointInfo(husky, j)[1].decode()
    if "wheel" in name:
        wheels[name] = j
print("wheels:", wheels)

# --- mount the Panda arm on top ---
base_pos, _ = p.getBasePositionAndOrientation(husky)
plate = [j for j in range(p.getNumJoints(husky))
         if p.getJointInfo(husky, j)[12].decode() == "top_plate_link"][0]
top_z = p.getAABB(husky, plate)[1][2]  # height of the Husky's top plate
print(f"top plate height: {top_z:.3f} m")
panda = p.loadURDF("franka_panda/panda.urdf", [base_pos[0] + 0.1, base_pos[1], top_z],
                   useFixedBase=False)
for j, q in enumerate(HOME):
    p.resetJointState(panda, j, q)
inertial_z = p.getDynamicsInfo(panda, -1)[3][2]
mount = p.createConstraint(husky, -1, panda, -1, p.JOINT_FIXED, [0, 0, 0],
                           [0.1, 0, top_z - base_pos[2] + inertial_z], [0, 0, 0])
p.changeConstraint(mount, maxForce=5000)
for hl in range(-1, p.getNumJoints(husky)):      # arm base must not collide with the robot
    for pl in (-1, 0):
        p.setCollisionFilterPair(husky, panda, hl, pl, 0)


def hold_arm():
    for j, q in enumerate(HOME):
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, q, force=240)


# --- Kiva warehouse shelf ---
shelf = p.loadSDF("kiva_shelf/model.sdf")[0]
p.resetBasePositionAndOrientation(shelf, [2.5, 0, 0], [0, 0, 0, 1])


def step(n):
    for _ in range(n):
        hold_arm()
        p.stepSimulation()
        time.sleep(1 / 240)


def drive(left, right, n):
    for name, j in wheels.items():
        v = left if "left" in name else right
        p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=v, force=100)
    step(n)


def pose():
    pos, orn = p.getBasePositionAndOrientation(husky)
    yaw = p.getEulerFromQuaternion(orn)[2]
    return pos[0], pos[1], yaw


step(240)  # settle
x0, y0, _ = pose()
print(f"start: x={x0:.3f} y={y0:.3f}")

drive(5, 5, 480)       # forward 2 seconds
drive(0, 0, 120)
x1, y1, _ = pose()
print(f"after forward: x={x1:.3f} y={y1:.3f}  moved {((x1-x0)**2 + (y1-y0)**2) ** 0.5:.3f} m")

drive(-3, 3, 240)      # turn in place 1 second
drive(0, 0, 120)
_, _, yaw = pose()
print(f"after turn: yaw={yaw * 57.3:.1f} deg")

arm_pos = p.getBasePositionAndOrientation(panda)[0]
print(f"arm base height: {arm_pos[2]:.3f} m (should stay on top of the Husky)")

while True:
    step(1)