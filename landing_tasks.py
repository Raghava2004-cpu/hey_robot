import math

from drone import Drone
from mobile_env import MobileEnv

RESULT = {}   # filled by landed_check, read by experiments


def landing_scene(wind=(0.8, 0.5), wheel_speed=1.2, drone_start=(-1.0, 1.0, 1.2)):
    def build(gui=False):
        env = MobileEnv(gui=gui)
        env.add_landing_pad()
        env.drone = Drone(env, start=drone_start, wind=wind)   # wind is hidden from the AI
        env.main_view = "landing"
        env.follow_camera = True
        env.step(120)
        env.set_wheels(wheel_speed, wheel_speed)               # the robot keeps driving forward
        return env
    return build


def landed_check(env):
    d = env.drone
    RESULT.clear()
    RESULT.update(landed=d.latched, miss=d.touchdown_miss,
                  ff=[float(d.ff[0]), float(d.ff[1])], wind=[float(d.wind[0]), float(d.wind[1])])
    if d.latched:
        return True, f"drone LANDED on the moving pad (touchdown {d.touchdown_miss * 100:.1f} cm from center)"
    if d.touchdown_miss is None:
        return False, "drone never tried to land"
    return False, f"drone did NOT land on the pad (at touchdown it was {d.touchdown_miss * 100:.0f} cm from the pad center)"


TEXT = "Land the drone on the orange landing pad on top of the moving ground robot. Wind may be blowing."
LANDING_TASKS = {
    "land_A": (TEXT, landing_scene(), landed_check),
    "land_B": (TEXT, landing_scene(wheel_speed=1.8, drone_start=(-1.5, -1.0, 1.2)), landed_check),
}