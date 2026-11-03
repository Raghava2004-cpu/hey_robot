import json
import math
import os
import re
import time

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from cube_tasks import BUILD_SPOT, TABLE_Z, get_font, word_slots
from mobile_env import DOWN, EE_LINK, H, W

CUBE_METHODS = {"look", "pick", "place", "is_holding", "table_height", "build_spot", "word_slots"}
MAX_ACTIONS = 40
SAFE_Z = 0.82                    # travel height: above any tower we will build
PARK = (0.30, -0.40)             # arm rests here, beside the robot: out of the camera's view
MAX_TOP = TABLE_Z + 0.28         # nothing real on the table is higher (5-block tower = 0.70)
TABLE_X0 = 0.62                  # near edge of the usable table
CUBE_DOC = """
The robot arm works at a table with 5 cm letter blocks (each block shows one capital letter).
robot.look() -> list of blocks the camera sees (SAM 3.1 finds them, Muse Spark reads the letters):
    [{"letter": "M", "pos": [x, y, z_top]}, ...]   z_top = height of the block's TOP face. letter is "?" if unreadable.
    For a tower, look() reports the TOP block of the tower. A block in the robot's own hand is not reported.
robot.pick(x, y, z_top) -> True/False   (grasp the block whose top face is at that point)
robot.place(x, y, z_surface) -> True/False   (put the held block down centred at (x, y), onto a surface whose top is at about z_surface: the table, or the top of another block. The arm lowers slowly and releases when the block touches something.)
robot.is_holding() -> True/False
robot.table_height() -> height of the table top in metres
robot.build_spot() -> [x, y] where a tower must be built
robot.word_slots(n) -> list of n [x, y] slots for a word, LEFT to RIGHT as the robot sees them
Blocks are exactly 0.05 m tall. Positions from look() have about 1 cm error. Blocks can move when you place others, so look() again when unsure.
Math helpers: math.sqrt, math.hypot, math.atan2, math.pi
"""


def read_prompt(inventory):
    known = f" The blocks only use these letters: {', '.join(inventory)}." if inventory else ""
    return ("Each numbered tile shows one block (or a tower of blocks) photographed at an angle. "
            "Read the single capital letter printed on the block in each tile; for a tower, read the TOP block. "
            "Letters may be rotated or upside down (an upside-down M looks like W)." + known +
            ' Reply ONLY with JSON like {"1": "M", "2": "E"} using the tile numbers as keys. '
            'Use "?" if unreadable.')


def parse_letters(text, n):
    """Forgiving reader: JSON {"1": "M"} / {"#1": "M"}, lines like "1: M", or a plain list "o block, a block, ..."."""
    text = text or ""
    out = ["?"] * n
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            for k, v in json.loads(m.group(0)).items():
                d = re.sub(r"\D", "", str(k))
                s = re.sub(r"[^A-Za-z]", "", str(v)).upper()
                if d and 1 <= int(d) <= n and s:
                    out[int(d) - 1] = s[0]
        except ValueError:
            pass
    if all(c == "?" for c in out):
        for d, s in re.findall(r"#?\s*(\d+)\s*[\"']?\s*[:=\-]\s*[\"']?([A-Za-z])\b", text):
            if 1 <= int(d) <= n:
                out[int(d) - 1] = s.upper()
    if all(c == "?" for c in out):                          # plain list, in tile order
        singles = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text)
        if len(singles) == n:
            out = [s.upper() for s in singles]
    return out


class CubeAPI:
    def __init__(self, env):
        self.env = env
        self.log = []
        self.last_reply = ""
        self.inventory = sorted(set(getattr(env, "letter_of", {}).values()))

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        self.env.say(f"{action} {arg}")
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")

    # ---------- safe arm motion: straight lines in space, never swinging low ----------
    def _ik(self, pos):
        env = self.env
        q = p.calculateInverseKinematics(env.panda, EE_LINK, list(pos), DOWN, maxNumIterations=50)
        env.arm_q = list(q[:7])

    def _go(self, target, step_len=0.01):
        env = self.env
        start = np.array(env.hand_pos())
        target = np.array(target, dtype=float)
        n = max(5, int(np.linalg.norm(target - start) / step_len))
        for k in range(1, n + 1):
            self._ik(start + (target - start) * k / n)
            env.step(3)
        env.move_arm_to(target.tolist(), steps=60)       # settle exactly on the target

    def _travel(self, x, y, z):
        """Lift to SAFE_Z, move across at SAFE_Z, then go straight down."""
        hx, hy, _ = self.env.hand_pos()
        self._go([hx, hy, SAFE_Z])
        self._go([x, y, SAFE_Z])
        self._go([x, y, z])

    def _park(self):
        """Go up, then rest beside the robot where the camera cannot see the arm."""
        hx, hy, _ = self.env.hand_pos()
        self._go([hx, hy, SAFE_Z])
        self._go([PARK[0], PARK[1], SAFE_Z])

    def _parked(self):
        hx, hy, hz = self.env.hand_pos()
        return math.hypot(hx - PARK[0], hy - PARK[1]) < 0.05 and abs(hz - SAFE_Z) < 0.05

    # ---------- perception: SAM 3.1 boxes + depth -> 3D, Muse Spark reads letters ----------
    def look(self):
        from perception import sam_find
        env = self.env
        if not self._parked():                              # never let the arm hide a block
            self._park()
        rgb, depth = env.camera("front")
        boxes = []
        for ph in ("block", "cube", "white block"):
            boxes = sam_find(rgb, ph)
            if boxes:
                break
        m_inv = np.linalg.inv(env._matrix())
        blocks = []
        for b in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b["box"]]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            us, vs = np.meshgrid(np.arange(x0, x1 + 1, 2), np.arange(y0, y1 + 1, 2))
            us, vs = us.ravel(), vs.ravel()
            d = depth[vs, us]
            ndc = np.stack([2 * us / W - 1, 1 - 2 * vs / H, 2 * d - 1, np.ones_like(d)])
            w = m_inv @ ndc
            pts = (w[:3] / w[3]).T
            pts = pts[pts[:, 2] > TABLE_Z + 0.01]          # drop table pixels
            if len(pts) < 5:
                continue
            ztop = pts[:, 2].max()
            top = pts[pts[:, 2] > ztop - 0.008]             # the top face
            pos = [round(float(np.median(top[:, 0])), 3), round(float(np.median(top[:, 1])), 3),
                   round(float(ztop), 3)]
            if pos[2] > MAX_TOP or pos[0] < TABLE_X0:       # the robot itself, or off the table
                continue
            dup = next((o for o in blocks if math.hypot(o["pos"][0] - pos[0], o["pos"][1] - pos[1]) < 0.03), None)
            if dup:
                if pos[2] > dup["pos"][2]:
                    dup.update(pos=pos, box=(x0, y0, x1, y1))
                continue
            blocks.append({"pos": pos, "box": (x0, y0, x1, y1)})
        blocks.sort(key=lambda o: -o["pos"][1])           # left to right as the robot sees it
        letters = self._read_letters(rgb, blocks)
        seen = [{"letter": L, "pos": o["pos"]} for o, L in zip(blocks, letters)]
        self._record("look", "", seen)
        return seen

    def _read_letters(self, rgb, blocks):
        if not blocks:
            return []
        from llm import spark_vision
        img = Image.fromarray(rgb)
        font = get_font(18)
        tiles = []
        for i, o in enumerate(blocks):
            x0, y0, x1, y1 = o["box"]
            crop = img.crop((max(x0 - 4, 0), max(y0 - 4, 0), min(x1 + 4, W), min(y1 + 4, H))).resize((120, 120))
            tile = Image.new("RGB", (140, 152), "white")
            tile.paste(crop, (10, 28))
            ImageDraw.Draw(tile).text((8, 3), f"#{i + 1}", fill=(220, 0, 0), font=font)
            tiles.append(tile)
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        mosaic = Image.new("RGB", (cols * 140, rows * 152), "white")
        for i, t in enumerate(tiles):
            mosaic.paste(t, ((i % cols) * 140, (i // cols) * 152))
        os.makedirs("runs", exist_ok=True)
        mosaic.save("runs/letters.png")
        prompt = read_prompt(self.inventory)
        letters, replies = ["?"] * len(blocks), []
        for k in range(3):                                  # up to 3 tries, short pause between
            if k:
                time.sleep(1.5)
            try:
                text = spark_vision(prompt, np.array(mosaic))
            except Exception as e:
                text = f"ERROR: {e}"
            replies.append(str(text))
            letters = parse_letters(str(text), len(blocks))
            if any(c != "?" for c in letters):
                break
        self.last_reply = replies[-1]
        with open("runs/letters_reply.txt", "w", encoding="utf-8") as f:
            f.write("\n\n----- next try -----\n\n".join(replies))
        return letters

    # ---------- actions ----------
    def _in_reach(self, x, y, z):
        return 0.55 < x < 0.95 and -0.35 < y < 0.35 and TABLE_Z - 0.02 < z < TABLE_Z + 0.4

    def pick(self, x, y, z_top):
        x, y, z = float(x), float(y), float(z_top)
        if not self._in_reach(x, y, z):
            self._record("pick", [x, y, z], "REFUSED: outside the work area")
            return False
        ok, msg = self._pick(x, y, z)
        self._record("pick", [round(x, 3), round(y, 3), round(z, 3)], msg)
        return ok

    def _pick(self, x, y, ztop):
        env = self.env
        if env.held:
            return False, "already holding a block"
        env.finger = 0.04
        self._travel(x, y, ztop + 0.08)
        self._go([x, y, ztop - 0.015])                     # straight down, fingers around the block
        best, best_d = env._closest_object_to_hand()
        if best is None or best_d > 0.03:
            self._park()
            return False, f"nothing graspable under the hand (closest {best_d:.3f} m)"
        env.finger = 0.0
        env.step(60)
        h = p.getLinkState(env.panda, EE_LINK)
        inv_pos, inv_orn = p.invertTransform(h[4], h[5])
        o_pos, o_orn = p.getBasePositionAndOrientation(env.objects[best])
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, o_pos, o_orn)
        c = p.createConstraint(env.panda, EE_LINK, env.objects[best], -1, p.JOINT_FIXED,
                               [0, 0, 0], rel_pos, [0, 0, 0], rel_orn)
        env.held = (best, c)
        env.finger = 0.024                                  # relax: fingers just touch the block
        env.step(30)
        self._park()
        return True, "picked a block"

    def place(self, x, y, z_surface):
        x, y, z = float(x), float(y), float(z_surface)
        if not self._in_reach(x, y, z):
            self._record("place", [x, y, z], "REFUSED: outside the work area")
            return False
        ok, msg = self._place(x, y, z)
        self._record("place", [round(x, 3), round(y, 3), round(z, 3)], msg)
        return ok

    def _place(self, x, y, z):
        """Guarded move: lower 2 mm at a time, stop as soon as the block touches something, release."""
        env = self.env
        if not env.held:
            return False, "not holding anything"
        name, c = env.held
        body = env.objects[name]
        self._travel(x, y, z + 0.12)
        hz, touched = z + 0.10, False
        while hz > z - 0.03:
            hz -= 0.002
            self._ik([x, y, hz])
            env.step(4)
            if any(cp[2] != env.panda for cp in p.getContactPoints(bodyA=body)):
                touched = True
                break
        env.step(30)                                        # let any swing die out
        p.removeConstraint(c)
        env.held = None
        env.finger = 0.04
        env.step(120)
        # release check (like a gripper's finger sensor): did the block come up with the hand?
        z_block = env.get_pos(name)[2]
        hx, hy, hz_now = env.hand_pos()
        self._go([x, y, hz_now + 0.04])
        if env.get_pos(name)[2] > z_block + 0.01:
            self._go([x, y, hz_now], step_len=0.005)
            env.step(120)
            self._go([x, y, hz_now + 0.04])
        self._park()
        return True, "placed the block" + ("" if touched else " (no contact felt, released at the lowest point)")

    def is_holding(self):
        return self.env.held is not None

    def table_height(self):
        return TABLE_Z

    def build_spot(self):
        return list(BUILD_SPOT)

    def word_slots(self, n):
        n = max(1, min(6, int(n)))
        return word_slots(n)