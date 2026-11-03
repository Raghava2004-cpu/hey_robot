"""Counts every Meta model call and the time it takes, so each job can print where its minutes went.

install() wraps SAM 3.1, Muse Spark / Wonder, Muse Image and Voice Transcribe. Call it BEFORE the modules
that use them are imported (command_center.py does this at start-up).
report() prints one line per job and returns the numbers (also appended to runs/timing.jsonl).
"""
import json
import os
import time

STATS = {}
_t0 = [time.time()]


def _add(name, seconds):
    s = STATS.setdefault(name, {"calls": 0, "seconds": 0.0})
    s["calls"] += 1
    s["seconds"] += seconds


def _wrap(mod, attr, name_of):
    fn = getattr(mod, attr, None)
    if fn is None or getattr(fn, "_metered", False):
        return

    def wrapped(*a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            name, dt = name_of(a, k), time.time() - t
            _add(name, dt)
            if name != "SAM 3.1":                      # SAM detections are already in the feed (muse_eye.log)
                try:
                    import muse_eye
                    tag = {"Muse Spark": "SPARK", "Muse Wonder": "WONDER", "Muse Image": "IMAGE",
                           "Voice Transcribe": "VOICE"}.get(name, name)
                    muse_eye.event(tag, f"{name} answered in {dt:.1f} s")
                except Exception:
                    pass
    wrapped._metered = True
    setattr(mod, attr, wrapped)


def install():
    try:
        import perception
        _wrap(perception, "sam_find", lambda a, k: "SAM 3.1")
    except Exception:
        pass
    try:
        import llm

        def who(a, k):
            model = k.get("model") or (a[2] if len(a) > 2 else "")
            return "Muse Wonder" if "wonder" in str(model) else "Muse Spark"
        _wrap(llm, "spark", who)
    except Exception:
        pass
    try:
        import imagine
        _wrap(imagine, "imagine", lambda a, k: "Muse Image")
    except Exception:
        pass
    try:
        import voice
        _wrap(voice, "transcribe", lambda a, k: "Voice Transcribe")
    except Exception:
        pass


def start():
    STATS.clear()
    _t0[0] = time.time()


def report(label=""):
    total = time.time() - _t0[0]
    models = sum(s["seconds"] for s in STATS.values())
    parts = [f"{n}: {s['calls']} calls, {s['seconds']:.0f} s" for n, s in sorted(STATS.items())]
    print(f"⏱️  TIMING {label}: {total:.0f} s total | " + " | ".join(parts)
          + f" | physics + robot motion + other: {max(0.0, total - models):.0f} s")
    rec = {"job": label, "total_s": round(total, 1),
           "models": {n: {"calls": s["calls"], "seconds": round(s["seconds"], 1)} for n, s in STATS.items()},
           "other_s": round(max(0.0, total - models), 1), "time": time.strftime("%Y-%m-%d %H:%M")}
    try:
        os.makedirs("runs", exist_ok=True)
        with open("runs/timing.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec
