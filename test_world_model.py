import math

from physics_model import predict_landing
from robot_env import TABLE_Z, RobotEnv

PYBULLET_DRAG = 0.3
TRUE_K = -math.log(1 - PYBULLET_DRAG)  # = 0.357
Z_LAND = TABLE_Z + 0.01 + 0.025       # ball center when resting on the tray floor

env = RobotEnv(gui=True, tray_xy=(1.05, 0.0))
env.add_ball("ball", [0.50, 0.15], [1, 0.5, 0, 1], drag=PYBULLET_DRAG)
env.step(120)
print(env.pick("ball"))  # test harness: simple pick, no camera needed

SPEED, ANGLE, HEADING = 2.208, 45, 0
print(env.toss(SPEED, ANGLE, HEADING))

# true landing = first moment the falling ball crosses Z_LAND
true = None
for i in range(1, len(env.last_flight)):
    z_prev, z = env.last_flight[i - 1][2], env.last_flight[i][2]
    if z < z_prev and z <= Z_LAND:
        true = env.last_flight[i]
        break
print(f"\nREAL landing (PyBullet): x={true[0]:.3f}  y={true[1]:.3f}")

models = [
    ("no drag            ", 0.0, 0.0),
    ("linear only        ", TRUE_K, 0.0),
    ("linear + quadratic ", PYBULLET_DRAG, PYBULLET_DRAG),
]
for label, k1, k2 in models:
    px, py, t = predict_landing(env.last_release, SPEED, ANGLE, HEADING, k1, k2, Z_LAND)
    err = math.hypot(px - true[0], py - true[1]) * 100
    print(f"{label} k1={k1:.3f} k2={k2:.3f}: x={px:.3f}  flight={t:.2f}s  error={err:.1f} cm")