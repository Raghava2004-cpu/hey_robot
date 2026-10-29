import ast

API_METHODS = {"see", "pick", "place_on", "is_holding", "put_aside", "ask",
               "see_far", "release_point", "toss"}
SAFE_BUILTINS = {"len", "range", "print", "abs", "min", "max", "round", "str", "int",
                 "float", "bool", "list", "enumerate", "sorted", "any", "all", "isinstance"}
MAX_ACTIONS = 30

API_DOC = """
robot.see(phrase) -> list of [x, y, z] positions (top surface, meters), best match first; [] if not seen
robot.pick(phrase) -> True/False   (pick up the object named by phrase)
robot.place_on(phrase) -> True/False   (put the held object on top of / into the named thing)
robot.is_holding() -> True/False
robot.put_aside() -> True/False   (put the held object down on a free spot of the table)
robot.ask(question) -> short text answer from looking at the camera now, e.g. robot.ask("Which object is on top of the purple cube?") -> "yellow cube"
robot.see_far(phrase) -> like see(), but uses a wide camera that also sees far things (far tray, thrown ball)
robot.release_point() -> [x, y, z] where a throw starts
robot.toss(speed, angle_deg, heading_deg) -> True/False   (throw the held object; heading 0 = +x, 90 = +y; speed 0-6 m/s; the landing spot is measured and logged)
Math helpers: math.sqrt, math.sin, math.cos, math.tan, math.atan2, math.radians, math.degrees, math.hypot, math.pi, and G = 9.81
Phrases are short noun phrases, e.g. "red cube", "blue cube", "tray".
"""


class RobotAPI:
    def __init__(self, env):
        self.env = env
        self.log = []
        self.last_phrase = None

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")

    def see(self, phrase):
        found = [[round(c, 3) for c in f["pos"]] for f in self.env.perceive(phrase)]
        self._record("see", phrase, found)
        return found

    def pick(self, phrase):
        ok, msg = self.env.pick_seen(phrase)
        self._record("pick", phrase, msg)
        if ok:
            self.last_phrase = phrase
        return ok

    def place_on(self, phrase):
        ok, msg = self.env.place_on(phrase)
        self._record("place_on", phrase, msg)
        return ok

    def is_holding(self):
        return self.env.held is not None

    def put_aside(self):
        ok, msg = self.env.put_aside()
        self._record("put_aside", "", msg)
        return ok

    def ask(self, question):
        from llm import spark_vision
        rgb, _ = self.env.camera()
        answer = spark_vision(question, rgb)
        self._record("ask", question, answer)
        return answer

    def see_far(self, phrase):
        found = [[round(c, 3) for c in f["pos"]] for f in self.env.perceive(phrase, "wide")]
        self._record("see_far", phrase, found)
        return found

    def release_point(self):
        from robot_env import RELEASE
        return list(RELEASE)

    def toss(self, speed, angle_deg, heading_deg):
        from robot_env import RELEASE
        speed, angle_deg, heading_deg = float(speed), float(angle_deg), float(heading_deg)
        args = f"speed={speed:.3f}, angle={angle_deg:.1f}, heading={heading_deg:.1f}"
        if not 0 < speed <= 6:
            self._record("toss", args, "REFUSED: speed outside safe range 0-6 m/s")
            return False
        ok, msg = self.env.toss(speed, angle_deg, heading_deg)
        if ok and self.last_phrase:  # measure where it landed (SAM 3.1, wide camera)
            found = self.env.perceive(self.last_phrase, "wide")
            if found:
                lx, ly, lz = found[0]["pos"]
                d = ((lx - RELEASE[0]) ** 2 + (ly - RELEASE[1]) ** 2) ** 0.5
                msg += f"; landed at [{lx:.3f}, {ly:.3f}, {lz:.3f}], {d:.3f} m from release point"
            else:
                msg += "; landing spot not visible"
        self._record("toss", args, msg)
        return ok


def check_code(code):
    """Return None if the code is safe, otherwise a reason string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(defs) != 1 or defs[0].name != "skill":
        return "code must be exactly one function named skill"
    banned = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda,
              ast.ClassDef, ast.With, ast.AsyncFunctionDef, ast.Await)
    for node in ast.walk(tree):
        if isinstance(node, banned):
            return f"not allowed: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"not allowed name: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return f"not allowed attribute: {node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id == "robot" \
                    and node.attr not in API_METHODS:
                return f"unknown robot method: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id not in SAFE_BUILTINS:
            return f"not allowed call: {node.func.id}"
    return None


def run_skill(code, api):
    import builtins
    import math
    from types import SimpleNamespace
    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    math_tools = SimpleNamespace(**{k: getattr(math, k) for k in
                                    ("sqrt", "sin", "cos", "tan", "atan", "atan2", "asin", "acos",
                                     "radians", "degrees", "hypot", "pi")})
    scope = {"__builtins__": safe, "math": math_tools, "G": 9.81}
    exec(compile(code, "<skill>", "exec"), scope)
    return scope["skill"](api)