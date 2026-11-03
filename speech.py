"""The robot talks back: every reply, key step and result is spoken out loud.

Windows: uses the voice built into Windows (System.Speech through PowerShell), so nothing to install.
If pyttsx3 is installed it is used instead. Elsewhere (or with --quiet) it only prints.

say("text")          queue a sentence (returns at once; the robot keeps working while it talks)
say("text", wait=True) speak and wait until it is finished (for the final result)
"""
import os
import queue
import shutil
import subprocess
import sys
import threading

QUIET = "--quiet" in sys.argv or os.environ.get("SKILLFORGE_QUIET") == "1"
MAX_WAITING = 2            # if the robot is faster than the voice, drop old step messages


class _Speaker:
    def __init__(self):
        self.q = queue.Queue()
        self.engine = None
        self.mode = "print"
        if not QUIET:
            try:
                import pyttsx3                       # optional
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 175)
                self.mode = "pyttsx3"
            except Exception:
                if sys.platform.startswith("win") and shutil.which("powershell"):
                    self.mode = "windows"
        self.busy = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _speak(self, text):
        if self.mode == "pyttsx3":
            self.engine.say(text)
            self.engine.runAndWait()
        elif self.mode == "windows":
            safe = text.replace("'", "''")
            cmd = ("Add-Type -AssemblyName System.Speech; "
                   "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; "
                   f"$s.Speak('{safe}')")
            subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _run(self):
        while True:
            text, done = self.q.get()
            self.busy.set()
            try:
                self._speak(text)
            except Exception:
                pass
            finally:
                self.busy.clear()
                if done is not None:
                    done.set()

    def say(self, text, wait=False, important=False):
        import home
        text = home.to_display(" ".join(str(text).split()))      # --home: "station C" -> "the bedside table"
        if not text:
            return
        print(f"🔊 {text}")
        if self.mode == "print":
            return
        if not wait and not important and self.q.qsize() >= MAX_WAITING:
            return                                   # keep up with the robot: skip a minor step message
        done = threading.Event() if wait else None
        self.q.put((text, done))
        if wait:
            done.wait(timeout=30)


_speaker = None


def say(text, wait=False, important=False):
    global _speaker
    if _speaker is None:
        _speaker = _Speaker()
    _speaker.say(text, wait=wait, important=important)


# ---------- short spoken lines for the robots' actions ----------
def _items(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return "nothing"
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def narrate(action, arg, result):
    """A short sentence for an action, or None if it is not worth saying."""
    r = str(result)
    if action == "drone_scan":
        seen = r.split("seen:")[-1].split(";")[0].strip(" []").replace("'", "")
        seen = [s.strip() for s in seen.split(",") if s.strip()]
        return f"Drone: on {arg} I see {_items(seen)}." if seen else f"Drone: {arg} is empty."
    if action == "pick":
        if r.startswith("picked"):
            return f"Got the {arg}."
        if "slipped" in r:
            return f"The {arg} slipped. Trying again."
        return f"I could not pick the {arg}."
    if action == "place_on":
        return f"Put it in the {arg}." if r.startswith("placed") else f"I could not place it on the {arg}."
    if action == "put_aside":
        return "Put it down on the table." if r.startswith("placed") else None
    if action == "drone_grab":
        return f"Drone: got the {arg}." if "grabbed" in r or r.startswith("picked") else None
    if action == "drone_drop_into":
        return f"Drone: dropped it into the {arg}." if "dropped" in r else None
    if action == "handoff_to_robot":
        return ("Hand-off done: the drone dropped it into my basket while I was driving."
                if not r.startswith(("FAILED", "REFUSED", "the drone")) else "The hand-off failed.")
    if action == "take_from_basket":
        return f"Took the {arg} out of my basket." if r.startswith("picked") else None
    return None
