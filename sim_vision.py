"""FAST STAND-IN for SAM 3.1 and Spark's product check, used ONLY by the big experiments (--fast).
It reads the simulator's own object mask instead of calling the API. Demos and --sam runs use the
real models. Results from this mode must be labeled "simulator mask (SAM stand-in)"."""
import sys
import types

import numpy as np
import pybullet as p

_state = {"env": None, "seg": None, "body": None}
_orig = p.getCameraImage


def _camera(*a, **k):
    out = _orig(*a, **k)
    _state["seg"] = np.reshape(out[4], (out[1], out[0]))
    return out


def _groups(env, phrase):
    import drone
    from mobile_env import ALIASES
    out = set()
    for name, body in env.objects.items():
        if phrase in [name] + ALIASES.get(name, []) + drone.ALIASES.get(name, []):
            out.add(tuple(range(body, body + 5)) if name in ("tray", "bin") else (body,))
    return out


def sam_find(rgb, phrase):
    env, seg = _state["env"], _state["seg"]
    if env is None or seg is None:
        return []
    boxes = []
    for grp in _groups(env, phrase):
        ys, xs = np.nonzero(np.isin(seg, grp))
        if len(xs) < 15:
            continue
        boxes.append({"box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                      "center": [(xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0],
                      "conf": min(0.99, 0.5 + len(xs) / 4000)})
        _state["body"] = grp[0]
    return boxes


def spark_vision(question, img):
    env = _state["env"]
    return next((n for n, b in env.objects.items() if b == _state["body"]), "other")


def install(env):
    """Call right after building a scene."""
    _state["env"] = env
    p.getCameraImage = _camera
    sys.modules["perception"] = types.SimpleNamespace(sam_find=sam_find)
    llm = types.ModuleType("llm")
    llm.spark_vision = spark_vision
    sys.modules["llm"] = llm
