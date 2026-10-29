from robot_env import RobotEnv


def in_tray(env, name):
    tray, pos = env.get_pos("tray"), env.get_pos(name)
    ok = abs(pos[0] - tray[0]) < 0.1 and abs(pos[1] - tray[1]) < 0.1
    return ok, f"{name} cube is in the tray" if ok else f"{name} cube is NOT in the tray"


def scene_blue_hidden(gui=True):
    env = RobotEnv(gui=gui)
    env.add_cube("blue", [0.58, 0.12], [0, 0, 1, 1])
    env.step(60)
    env.add_cube_on("red", "blue", [1, 0, 0, 1])
    env.step(120)
    return env


def scene_purple_hidden(gui=True):
    env = RobotEnv(gui=gui)
    env.add_cube("red", [0.50, 0.22], [1, 0, 0, 1])          # distractor
    env.add_cube("purple", [0.66, 0.05], [0.5, 0, 0.8, 1])
    env.step(60)
    env.add_cube_on("yellow", "purple", [1, 0.9, 0, 1])      # yellow hides purple
    env.step(120)
    return env


TASKS = {
    "blue_hidden": ("Put the blue cube into the tray.", scene_blue_hidden,
                    lambda env: in_tray(env, "blue")),
    "purple_hidden": ("Put the purple cube into the tray.", scene_purple_hidden,
                      lambda env: in_tray(env, "purple")),
}
import random

COLORS = {"red": [1, 0, 0, 1], "blue": [0, 0, 1, 1], "purple": [0.5, 0, 0.8, 1],
          "yellow": [1, 0.9, 0, 1], "orange": [1, 0.5, 0, 1]}


def random_hidden_task(seed):
    rng = random.Random(seed)
    target, cover, distractor = rng.sample(list(COLORS), 3)

    def spot():
        return [rng.uniform(0.48, 0.70), rng.uniform(0.02, 0.28)]

    a = spot()
    while True:
        b = spot()
        if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > 0.12:
            break

    def build(gui=False):
        env = RobotEnv(gui=gui)
        env.add_cube(distractor, b, COLORS[distractor])
        env.add_cube(target, a, COLORS[target])
        env.step(60)
        env.add_cube_on(cover, target, COLORS[cover])
        env.step(120)
        return env

    task = f"Put the {target} cube into the tray."
    return task, build, (lambda env: in_tray(env, target))