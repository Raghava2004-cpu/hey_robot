"""Home mode (--home): the same robot team and the same skills, in a home with two rooms and people.

Why: the warehouse shows the robots work; the home shows who they help. A grandmother on the sofa (or
someone who cannot walk far or reach high) says "Hey robot, bring me the soup can from the storeroom" or
"get the strawberry from the top kitchen shelf", and the robot and the drone do it.

How (honest): the four living-room work spots keep their warehouse names inside, so every tested move,
grasp, camera and check is reused unchanged. Home mode adds one new work spot, the storeroom table, in a
second room behind a door:
    station A       = kitchen counter
    station B       = dining table
    station C       = bedside table
    station D       = storeroom table (second room; home mode only)
    packing station = sofa table (with the serving tray)
    top shelf       = top kitchen shelf
The spoken order is translated to the inside names before routing; what the robot says, the place labels
and the Muse Brain panel are translated back to home names.
Furniture, walls and people are visual only (no collision shapes): they can never block the robot or the
drone. The storeroom sits on the diagonal through the gap between the dining table and the bedside table,
so the robot's normal straight-line driving goes through the door.

Turn on:  python command_center.py --home ...   (or set HEYROBOT_HOME=1). Off by default.
"""
import math
import os
import re

HOME_NAMES = {"station A": "kitchen counter",
              "station B": "dining table",
              "station C": "bedside table",
              "station D": "storeroom table",
              "packing station": "sofa table"}
LEGEND = ["HOME: station A=kitchen counter  B=dining table",
          "      C=bedside table  D=storeroom  packing=sofa table"]

# the second room: a storeroom in the back-left corner, door in its front wall
STORE_CENTER = (-3.6, 3.6)             # the storeroom table
STORE_YAW = 3 * math.pi / 4            # the robot faces it along the diagonal (parks at -3.0, 3.0)
STORE_WALL_Y, STORE_WALL_X = 2.4, -1.8
DOOR = (-3.3, -1.8)                    # door opening (x range) in the wall at y = 2.4
LAYOUT = {"soup can": ("station D", 1),    # in the home the soup can is kept in the storeroom
          "water bottle": ("station A", 1), "notebook": ("station C", 2), "pen": ("station D", 2)}

# home things that are not in the YCB model set: simple shapes, sized for the Panda gripper (opens 8 cm)
MADE = ["water bottle", "notebook", "pen"]
ALIASES = {"water bottle": ["water bottle", "plastic bottle", "small bottle", "bottle"],
           "notebook": ["notebook", "book", "small book", "blue book"],
           "pen": ["pen", "marker pen", "marker"]}

TABLE = (0.36, 0.24, 0.15, 1)          # dark walnut: items stand out like on the grey warehouse tables
SOFA_TABLE = (0.50, 0.36, 0.24, 1)
SHELF = (0.62, 0.45, 0.30, 1)          # the tall kitchen shelf (the drone's "top shelf")


def active():
    return os.environ.get("HEYROBOT_HOME") == "1"


def places():
    """Work spots with a table, in this mode."""
    return ["station A", "station B", "station C"] + (["station D"] if active() else [])


# ---------- names ----------
def to_internal(order):
    """Spoken home words -> the inside place names the skills know. 'the bedside table' -> 'station C'."""
    s = str(order)
    rules = [(r"\b(the )?(top|high|upper) (kitchen )?(shelf|cupboard|cabinet)\b", "the top shelf"),
             (r"\b(the )?kitchen (shelf|cupboard|cabinet)\b", "the top shelf"),
             (r"\b(the )?(serving )?tray\b", "the tray"),
             (r"\b(the )?(store ?room|storage( room)?|pantry|other room|back room)( table)?\b", "station D"),
             (r"\b(the )?kitchen( counter| table| worktop)?\b", "station A"),
             (r"\b(the )?dining( room)?( table)?\b", "station B"),
             (r"\b(the )?(bedside( table)?|night ?stand|bedroom|bed)\b", "station C"),
             (r"\b(the )?((sofa|couch|living room)( table)?|coffee table)\b", "the packing station"),
             (r"\bstation ([A-D])( table)+\b", r"station \1")]          # "bedside table table" (a mishearing)
    for pat, rep in rules:
        s = re.sub(pat, rep, s, flags=re.I)
    return s


def to_display(text):
    """Inside names -> home names, for what people see and hear. Safe to run twice."""
    s = str(text)
    if not active():
        return s
    for k in "ABCD":
        name = HOME_NAMES[f"station {k}"]
        s = re.sub(rf"\bstation {k}'s table\b", f"the {name}", s, flags=re.I)
        s = re.sub(rf"\b(the )?table of station {k}\b", f"the {name}", s, flags=re.I)
        s = re.sub(rf"\bstation {k}\b", f"the {name}", s, flags=re.I)
    s = re.sub(r"\b(the )?packing (station|table)\b", "the sofa table", s, flags=re.I)
    s = re.sub(r"\btop shelf\b", "top kitchen shelf", s, flags=re.I)
    s = re.sub(r"(?<!serving )\btray\b", "serving tray", s)
    s = re.sub(r"\bwarehouse\b", "home", s)
    s = re.sub(r"\bWarehouse\b", "Home", s)
    s = s.replace("dark grey table", "dark wooden table")
    s = re.sub(r"\b([Tt]he) the\b", r"\1", s)
    s = re.sub(r"\b(on|to|at|from|in) the the\b", r"\1 the", s)
    return s


def label(name):
    """Name for a floating place label."""
    return to_display(name).replace("the ", "", 1) if active() else name


def table_color(name, rgba):
    if not active():
        return rgba
    if name == "packing station":
        return SOFA_TABLE
    return TABLE if name in HOME_NAMES else rgba


# ---------- the rooms (visual only) ----------
def items():
    """Extra product names in this mode."""
    return list(MADE) if active() else []


def make_item(p, name, x, y, z_table, yaw):
    """Build one home thing on the table at (x, y). yaw = the station's direction (robot -> table).
    Sizes: water bottle 5.5 cm wide, 11 cm tall (like the soup can); notebook 7.2 x 10.5 x 2.4 cm (like the
    gelatin box; a normal book is wider than the open gripper); a thick marker pen 11 cm x 2.4 cm, pointing at
    the robot so the fingers close across it (a thinner pen sits lower than the fingertips can reach)."""
    if name == "water bottle":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.0275, height=0.11)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0275, length=0.09, rgbaColor=(0.55, 0.80, 0.97, 1),
                                  visualFramePosition=[0, 0, -0.01])
        neck = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.02, rgbaColor=(0.55, 0.80, 0.97, 1))
        cap = p.createVisualShape(p.GEOM_CYLINDER, radius=0.016, length=0.012, rgbaColor=(0.10, 0.35, 0.85, 1))
        label = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0280, length=0.035, rgbaColor=(0.97, 0.97, 0.97, 1))
        links = [(neck, [0, 0, 0.045]), (cap, [0, 0, 0.059]), (label, [0, 0, -0.01])]
        body = p.createMultiBody(0.25, col, vis, [x, y, z_table + 0.056],
                                 linkMasses=[0] * 3, linkCollisionShapeIndices=[-1] * 3,
                                 linkVisualShapeIndices=[v for v, _ in links],
                                 linkPositions=[q for _, q in links], linkOrientations=[[0, 0, 0, 1]] * 3,
                                 linkInertialFramePositions=[[0, 0, 0]] * 3,
                                 linkInertialFrameOrientations=[[0, 0, 0, 1]] * 3,
                                 linkParentIndices=[0] * 3, linkJointTypes=[p.JOINT_FIXED] * 3,
                                 linkJointAxis=[[0, 0, 1]] * 3)
    elif name == "notebook":
        half = [0.036, 0.0525, 0.012]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=(0.12, 0.22, 0.55, 1))
        pages = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.0345, 0.0515, 0.009], rgbaColor=(0.97, 0.96, 0.90, 1))
        strip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.037, 0.008, 0.0125], rgbaColor=(0.95, 0.80, 0.20, 1))
        links = [(pages, [0.002, 0.0015, 0.0]), (strip, [0, -0.030, 0.0])]
        # the 7.2 cm side runs sideways (the fingers close sideways), the long side points at the robot
        orn = p.getQuaternionFromEuler([0, 0, yaw + math.pi / 2])
        body = p.createMultiBody(0.15, col, vis, [x, y, z_table + 0.013], orn,
                                 linkMasses=[0] * 2, linkCollisionShapeIndices=[-1] * 2,
                                 linkVisualShapeIndices=[v for v, _ in links],
                                 linkPositions=[q for _, q in links], linkOrientations=[[0, 0, 0, 1]] * 2,
                                 linkInertialFramePositions=[[0, 0, 0]] * 2,
                                 linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
                                 linkParentIndices=[0] * 2, linkJointTypes=[p.JOINT_FIXED] * 2,
                                 linkJointAxis=[[0, 0, 1]] * 2)
    elif name == "pen":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.012, height=0.11)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.012, length=0.085, rgbaColor=(0.15, 0.30, 0.80, 1),
                                  visualFramePosition=[0, 0, -0.0125])
        cap = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0125, length=0.025, rgbaColor=(0.08, 0.08, 0.10, 1))
        clip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.0015, 0.011], rgbaColor=(0.75, 0.75, 0.78, 1))
        links = [(cap, [0, 0, 0.0425]), (clip, [0.0135, 0, 0.040])]
        # lying down, pointing at the robot (its axis along the station direction)
        orn = p.getQuaternionFromEuler([0, math.pi / 2, yaw])
        body = p.createMultiBody(0.02, col, vis, [x, y, z_table + 0.0125], orn,
                                 linkMasses=[0] * 2, linkCollisionShapeIndices=[-1] * 2,
                                 linkVisualShapeIndices=[v for v, _ in links],
                                 linkPositions=[q for _, q in links], linkOrientations=[[0, 0, 0, 1]] * 2,
                                 linkInertialFramePositions=[[0, 0, 0]] * 2,
                                 linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
                                 linkParentIndices=[0] * 2, linkJointTypes=[p.JOINT_FIXED] * 2,
                                 linkJointAxis=[[0, 0, 1]] * 2)
    else:
        raise ValueError(name)
    p.changeDynamics(body, -1, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.01)
    return body


def _box(p, half, pos, rgba, yaw=0.0, pitch=0.0):
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(0, -1, v, pos, p.getQuaternionFromEuler([0, pitch, yaw]))


def _cyl(p, r, h, pos, rgba):
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=rgba)
    return p.createMultiBody(0, -1, v, pos)


def _ball(p, r, pos, rgba):
    v = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba)
    return p.createMultiBody(0, -1, v, pos)


SKIN = (0.87, 0.70, 0.58, 1)
HAIR = (0.85, 0.85, 0.85, 1)           # grey hair


def _person_on_sofa(p, x, y, top):
    """A grandmother sitting on the sofa, facing the room (+y). No red or green clothes (SAM looks for fruit)."""
    shirt, pants, shoes = (0.45, 0.36, 0.62, 1), (0.32, 0.33, 0.38, 1), (0.15, 0.12, 0.10, 1)
    for dx in (-0.1, 0.1):
        _box(p, [0.08, 0.22, 0.07], [x + dx, y + 0.20, top + 0.07], pants)          # thighs
        _box(p, [0.06, 0.06, 0.24], [x + dx, y + 0.40, top - 0.17], pants)          # lower legs
        _box(p, [0.06, 0.11, 0.04], [x + dx, y + 0.45, 0.04], shoes)
    _box(p, [0.18, 0.11, 0.27], [x, y - 0.04, top + 0.40], shirt, pitch=0.0)        # body
    for dx in (-0.235, 0.235):
        _box(p, [0.055, 0.06, 0.16], [x + dx, y - 0.02, top + 0.47], shirt)        # upper arms
        _box(p, [0.05, 0.16, 0.05], [x + dx * 0.8, y + 0.14, top + 0.30], shirt)   # forearms on the lap
        _ball(p, 0.045, [x + dx * 0.75, y + 0.31, top + 0.30], SKIN)               # hands
    _cyl(p, 0.05, 0.08, [x, y - 0.04, top + 0.71], SKIN)                           # neck
    _ball(p, 0.11, [x, y - 0.04, top + 0.84], SKIN)                                # head
    _ball(p, 0.112, [x, y - 0.075, top + 0.87], HAIR)                              # hair (back of the head)
    _ball(p, 0.06, [x, y - 0.14, top + 0.93], HAIR)                                # hair bun
    for dx in (-0.04, 0.04):
        _ball(p, 0.012, [x + dx, y + 0.065, top + 0.86], (0.15, 0.15, 0.2, 1))     # eyes
    _box(p, [0.075, 0.01, 0.025], [x, y + 0.095, top + 0.86], (0.2, 0.2, 0.25, 1))  # glasses


def _person_in_bed(p):
    """Grandfather resting in bed, under the blanket."""
    _box(p, [0.55, 0.22, 0.09], [-2.80, 0.0, 0.64], (0.30, 0.45, 0.70, 1))         # body under the blanket
    _box(p, [0.10, 0.10, 0.06], [-2.28, 0.15, 0.66], (0.30, 0.45, 0.70, 1))        # feet bump
    _box(p, [0.10, 0.10, 0.06], [-2.28, -0.15, 0.66], (0.30, 0.45, 0.70, 1))
    _ball(p, 0.10, [-3.33, 0.0, 0.76], SKIN)                                       # head on the pillow
    _ball(p, 0.102, [-3.37, 0.0, 0.78], HAIR)
    _box(p, [0.05, 0.25, 0.04], [-3.05, 0.0, 0.72], (0.80, 0.78, 0.70, 1))         # arms over the blanket
    _ball(p, 0.04, [-3.05, 0.27, 0.72], SKIN)
    _ball(p, 0.04, [-3.05, -0.27, 0.72], SKIN)


def build_room(p):
    """Floor, low 'doll-house' walls (so every camera sees in), rug, kitchen, dining chairs, bed, sofa, plants,
    two people, and a storeroom (second room with a door). The work area stays clear."""
    plane = p.loadURDF("plane.urdf")                          # physics floor (invisible)
    p.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0])
    _box(p, [8, 8, 0.002], [0, 0, -0.002], (0.86, 0.82, 0.74, 1))              # light floor: the orange pad stays clear
    for k in range(-9, 10):                                                     # floor boards
        _box(p, [4.75, 0.004, 0.0005], [0, k * 0.5, 0.0003], (0.78, 0.73, 0.64, 1))
    _box(p, [1.25, 1.25, 0.001], [0, 0, 0.0008], (0.64, 0.70, 0.76, 1))         # rug in the middle
    _box(p, [1.30, 1.30, 0.0007], [0, 0, 0.0005], (0.38, 0.45, 0.55, 1))        # rug border
    wall, skirt = (0.93, 0.89, 0.80, 1), (0.97, 0.97, 0.95, 1)
    # outer walls (1 m), cream, with a white skirting board and two windows
    for y in (-4.8, 4.8):
        _box(p, [4.8, 0.05, 0.5], [0, y, 0.5], wall)
        _box(p, [4.8, 0.06, 0.06], [0, y * 0.995, 0.06], skirt)
    for x in (-4.8, 4.8):
        _box(p, [0.05, 4.8, 0.5], [x, 0, 0.5], (0.90, 0.86, 0.77, 1))
        _box(p, [0.06, 4.8, 0.06], [x * 0.995, 0, 0.06], skirt)
    for x in (0.0, 2.6):
        _box(p, [0.7, 0.06, 0.25], [x, 4.78, 0.62], (0.62, 0.80, 0.95, 1))     # window glass
        _box(p, [0.74, 0.065, 0.02], [x, 4.78, 0.36], skirt)                   # sill
    # ---- the storeroom: walls with a door, a different floor, shelves and a washing machine ----
    wy, wx = STORE_WALL_Y, STORE_WALL_X
    left = (-4.8 + DOOR[0]) / 2
    _box(p, [(DOOR[0] + 4.8) / 2, 0.05, 0.5], [left, wy, 0.5], wall)           # front wall, left of the door
    _box(p, [0.05, (4.8 - wy) / 2, 0.5], [wx, (wy + 4.8) / 2, 0.5], wall)      # side wall
    _box(p, [0.05, 0.05, 0.55], [DOOR[0], wy, 0.55], (0.55, 0.40, 0.28, 1))    # door frame posts
    _box(p, [0.05, 0.05, 0.55], [DOOR[1], wy, 0.55], (0.55, 0.40, 0.28, 1))
    _box(p, [0.38, 0.025, 0.46], [DOOR[0] - 0.40, wy + 0.08, 0.48], (0.70, 0.52, 0.36, 1))  # open door, against the wall
    _box(p, [(4.8 + wx) / 2 - 0.05, (4.8 - wy) / 2 - 0.05, 0.001],
         [(-4.8 + wx) / 2, (wy + 4.8) / 2, 0.001], (0.74, 0.76, 0.78, 1))     # grey tiles
    for sx in (-4.55,):                                                        # metal shelving on the back-left wall
        for z in (0.35, 0.75, 1.15):
            _box(p, [0.20, 0.70, 0.015], [sx, 3.15, z], (0.70, 0.72, 0.75, 1))
        for dy in (-0.68, 0.68):
            _box(p, [0.02, 0.02, 0.62], [sx, 3.15 + dy, 0.62], (0.45, 0.47, 0.50, 1))
    _box(p, [0.30, 0.30, 0.42], [-2.25, 4.40, 0.42], (0.96, 0.96, 0.97, 1))    # washing machine
    _cyl(p, 0.17, 0.01, [-2.25, 4.095, 0.48], (0.55, 0.65, 0.75, 1))
    # kitchen behind the kitchen counter (station A, +x): cupboards, worktop, sink, fridge
    _box(p, [0.30, 0.80, 0.42], [2.55, 0.70, 0.42], (0.93, 0.93, 0.90, 1))
    _box(p, [0.32, 0.82, 0.025], [2.55, 0.70, 0.865], (0.25, 0.25, 0.27, 1))
    _box(p, [0.16, 0.20, 0.02], [2.55, 0.55, 0.88], (0.70, 0.72, 0.75, 1))     # sink
    for y in (0.15, 0.70, 1.25):
        _box(p, [0.005, 0.24, 0.30], [2.245, y, 0.42], (0.85, 0.85, 0.82, 1))  # cupboard doors
    _box(p, [0.36, 0.36, 0.92], [2.55, 1.95, 0.92], (0.95, 0.96, 0.97, 1))     # fridge
    _box(p, [0.005, 0.02, 0.25], [2.185, 1.75, 1.20], (0.6, 0.6, 0.62, 1))     # fridge handle
    # dining chairs behind the dining table (station B, +y)
    for x in (-0.38, 0.38):
        _box(p, [0.19, 0.19, 0.02], [x, 2.15, 0.45], (0.45, 0.30, 0.18, 1))
        _box(p, [0.19, 0.02, 0.25], [x, 2.33, 0.72], (0.45, 0.30, 0.18, 1))
        for dx in (-0.16, 0.16):
            for dy in (-0.16, 0.16):
                _box(p, [0.015, 0.015, 0.22], [x + dx, 2.15 + dy, 0.22], (0.40, 0.27, 0.16, 1))
    # bed behind the bedside table (station C, -x), head against the wall
    _box(p, [0.75, 1.00, 0.20], [-2.85, 0.0, 0.20], (0.42, 0.30, 0.20, 1))     # frame
    _box(p, [0.73, 0.97, 0.09], [-2.85, 0.0, 0.48], (0.96, 0.96, 0.94, 1))     # mattress
    _box(p, [0.55, 0.98, 0.03], [-2.65, 0.0, 0.58], (0.30, 0.45, 0.70, 1))     # blanket
    _box(p, [0.15, 0.70, 0.06], [-3.40, 0.0, 0.62], (1.0, 1.0, 1.0, 1))        # pillow
    _box(p, [0.06, 1.00, 0.45], [-3.62, 0.0, 0.45], (0.42, 0.30, 0.20, 1))     # headboard
    _person_in_bed(p)
    # sofa behind the sofa table (packing station, -y), facing the room, with grandma on it
    blue = (0.25, 0.42, 0.55, 1)
    _box(p, [1.00, 0.40, 0.22], [0, -2.62, 0.22], blue)
    _box(p, [1.00, 0.12, 0.38], [0, -2.95, 0.55], blue)
    for x in (-1.0, 1.0):
        _box(p, [0.10, 0.42, 0.32], [x, -2.62, 0.32], blue)
    _box(p, [0.40, 0.30, 0.06], [0.55, -2.55, 0.50], (0.30, 0.50, 0.64, 1))    # cushion
    _person_on_sofa(p, -0.25, -2.60, 0.48)
    # plants in two corners (the third corner is the storeroom)
    for x, y in ((3.9, -3.9), (3.9, 3.9)):
        _cyl(p, 0.18, 0.40, [x, y, 0.20], (0.75, 0.45, 0.30, 1))
        _ball(p, 0.35, [x, y, 0.70], (0.25, 0.55, 0.28, 1))
