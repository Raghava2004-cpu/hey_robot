import math
import os
import random

import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from mobile_env import MobileEnv

TABLE_Z = 0.45              # table top height
CUBE = 0.05                 # block size
BUILD_SPOT = (0.79, 0.0)    # far side of the table: away from the parked hand
ROW_X = 0.79                # the word row, also on the far side
SLOT_GAP = 0.09
LETTERS = ["M", "E", "T", "A", "O", "X"]    # O and X are distractors
# blocks start near the robot, in a zig-zag
GRID = [(0.66 if i % 2 == 0 else 0.71, -0.20 + 0.08 * i) for i in range(6)]
TEX_DIR = os.path.join("assets", "letters")


def get_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def letter_texture(letter):
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, f"{letter}.png")
    if not os.path.exists(path):
        img = Image.new("RGB", (128, 128), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([3, 3, 124, 124], outline=(40, 40, 40), width=6)
        font = get_font(96)
        box = d.textbbox((0, 0), letter, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        d.text(((128 - w) / 2 - box[0], (128 - h) / 2 - box[1]), letter, fill="black", font=font)
        img.save(path)
    return path


def word_slots(n):
    """n slots in a row, LEFT to RIGHT as the robot sees them (robot faces +x, so left is +y)."""
    start = (n - 1) / 2 * SLOT_GAP
    return [[ROW_X, round(start - i * SLOT_GAP, 3)] for i in range(n)]


def cube_scene(seed, letters=LETTERS):
    def build(gui=False):
        rng = random.Random(seed)
        env = MobileEnv(gui=gui)
        env.add_table("workbench", (0.85, 0.0), size=(0.5, 0.8))
        env.add_location("workbench", (0.0, 0.0, 0.0), [])
        spots = GRID[:]
        rng.shuffle(spots)
        env.letter_of = {}
        for L, (x, y) in zip(letters, spots):
            x += rng.uniform(-0.008, 0.008)
            y += rng.uniform(-0.008, 0.008)
            orn = p.getQuaternionFromEuler([0, 0, rng.uniform(-0.4, 0.4)])
            body = p.loadURDF("cube_small.urdf", [x, y, TABLE_Z + 0.03], orn)
            tex = p.loadTexture(letter_texture(L))
            p.changeVisualShape(body, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1])
            env.objects[f"block {L}"] = body
            env.letter_of[f"block {L}"] = L
        env.main_view = "front"
        if gui:
            p.resetDebugVisualizerCamera(1.4, 55, -35, [0.75, 0.0, 0.45])
        env.step(240)
        return env
    return build


# ---------- independent checkers (simulator ground truth, never shown to the robot) ----------
def _cubes(env):
    return {n: env.get_pos(n) for n in env.letter_of}


def tower_height(env, spot=BUILD_SPOT):
    near = sorted(z for (x, y, z) in _cubes(env).values()
                  if math.hypot(x - spot[0], y - spot[1]) < 0.035)
    h = 0
    for z in near:
        if abs(z - (TABLE_Z + CUBE / 2 + h * CUBE)) < 0.015:
            h += 1
    return h


def stack_check(target=4):
    def check(env):
        h = tower_height(env)
        return h >= target, f"tower height at the build spot: {h} blocks (goal: at least {target})"
    return check


def spell_check(word):
    def check(env):
        cubes = _cubes(env)
        slots = word_slots(len(word))
        good, notes = 0, []
        for L, (sx, sy) in zip(word, slots):
            x, y, z = cubes[f"block {L}"]
            ok = math.hypot(x - sx, y - sy) < 0.03 and z < TABLE_Z + CUBE
            good += ok
            notes.append(f"{L}:{'ok' if ok else 'wrong'}")
        intruders = [n for n, (x, y, z) in cubes.items() if env.letter_of[n] not in word
                     and any(math.hypot(x - sx, y - sy) < 0.03 for sx, sy in slots)]
        ok = good == len(word) and not intruders
        msg = f"word {word}: {' '.join(notes)}"
        if intruders:
            msg += f"; wrong blocks in the row: {intruders}"
        return ok, msg
    return check


STACK_TASK = ("Build a tower of at least 4 letter blocks stacked on top of each other at the build spot "
              "(robot.build_spot()). The letters do not matter.")


def spell_task(word):
    return (f"Spell the word {word}: put the letter blocks {', '.join(word)} into the word slots "
            f"(robot.word_slots({len(word)})), LEFT to RIGHT, one block per slot, each standing "
            f"directly on the table.")