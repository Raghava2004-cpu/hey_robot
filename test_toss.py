import math

from robot_env import RELEASE, RobotEnv

G = 9.81
HIDDEN_DRAG = 0.3  # the robot does NOT know this

env = RobotEnv(gui=True, tray_xy=(1.05, 0.0))   # tray far away, out of reach
env.add_ball("ball", [0.50, 0.15], [1, 0.5, 0, 1], drag=HIDDEN_DRAG)
env.step(120)

print("pick:", env.pick_seen("orange ball"))

# where is the tray? (measured by SAM 3.1 from the wide camera)
tx, ty, tz = env.perceive("tray", "wide")[0]["pos"]
tz += 0.025                                       # ball center height when resting in the tray
rx, ry, rz = RELEASE
d = math.hypot(tx - rx, ty - ry)                  # horizontal distance
h = tz - rz                                       # height change (negative = downhill)
angle = math.radians(45)

# projectile motion, no drag:  v^2 = g d^2 / (2 cos^2(a) (d tan(a) - h))
v = math.sqrt(G * d * d / (2 * math.cos(angle) ** 2 * (d * math.tan(angle) - h)))
heading = math.degrees(math.atan2(ty - ry, tx - rx))
print(f"target distance {d:.3f} m, height change {h:.3f} m -> pure-math speed {v:.3f} m/s")

print("toss:", env.toss(v, 45, heading))

lx, ly, _ = env.perceive("orange ball", "wide")[0]["pos"]
landed = math.hypot(lx - rx, ly - ry)
print(f"ball landed at {landed:.3f} m  (wanted {d:.3f} m)  -> {100 * (landed - d) / d:+.0f}%")

tray, ball = env.get_pos("tray"), env.get_pos("ball")
print("in tray:", abs(ball[0] - tray[0]) < 0.1 and abs(ball[1] - tray[1]) < 0.1)

while True:
    env.step()