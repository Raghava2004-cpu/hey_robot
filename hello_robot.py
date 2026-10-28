import time
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.5, 0, 0])
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.625], useFixedBase=True)
cube = p.loadURDF("cube_small.urdf", [0.6, 0.1, 0.65])
p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])
p.loadURDF("tray/traybox.urdf", [0.6, -0.25, 0.625], globalScaling=0.6)

EE_LINK = 11  # Panda hand link

def move_to(target, steps=240):
    for _ in range(steps):
        joints = p.calculateInverseKinematics(robot, EE_LINK, target,
                                              p.getQuaternionFromEuler([3.14, 0, 0]))
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joints[j], force=200)
        p.stepSimulation()
        time.sleep(1 / 240)

# Move above the cube, then above the tray
move_to([0.6, 0.1, 0.85])
move_to([0.6, -0.25, 0.85])
print("Arm moved. Close the window to exit.")

while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)