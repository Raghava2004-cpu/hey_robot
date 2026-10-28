import ast

API_METHODS = {"see", "pick", "place_on", "is_holding", "put_aside", "ask"}
SAFE_BUILTINS = {"len", "range", "print", "abs", "min", "max", "round"}
MAX_ACTIONS = 30

API_DOC = """
robot.ask(question) -> short text answer from looking at the camera now, e.g. robot.ask("Which object is on top of the purple cube?") -> "yellow cube"
robot.put_aside() -> True/False   (put the held object down on a free spot of the table)
robot.see(phrase) -> list of [x, y, z] positions (top surface, meters), best match first; [] if not seen
robot.pick(phrase) -> True/False   (pick up the object named by phrase)
robot.place_on(phrase) -> True/False   (put the held object on top of / into the named thing)
robot.is_holding() -> True/False
Phrases are short noun phrases, e.g. "red cube", "blue cube", "tray".
"""


class RobotAPI:
    def __init__(self, env):
        self.env = env
        self.log = []

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")
        
    def ask(self, question):
        from llm import spark_vision
        rgb, _ = self.env.camera()
        answer = spark_vision(question, rgb)
        self._record("ask", question, answer)
        return answer  
      
    def put_aside(self):
        ok, msg = self.env.put_aside()
        self._record("put_aside", "", msg)
        return ok    

    def see(self, phrase):
        found = [[round(c, 3) for c in f["pos"]] for f in self.env.perceive(phrase)]
        self._record("see", phrase, found)
        return found

    def pick(self, phrase):
        ok, msg = self.env.pick_seen(phrase)
        self._record("pick", phrase, msg)
        return ok

    def place_on(self, phrase):
        ok, msg = self.env.place_on(phrase)
        self._record("place_on", phrase, msg)
        return ok

    def is_holding(self):
        return self.env.held is not None


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
    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    scope = {"__builtins__": safe}
    exec(compile(code, "<skill>", "exec"), scope)
    return scope["skill"](api)