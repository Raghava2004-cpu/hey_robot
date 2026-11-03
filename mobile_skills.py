import json
import os
import re

MOBILE_METHODS = {"places", "go_to", "where_am_i", "see", "pick", "place_on", "put_aside",
                  "is_holding", "ask", "look_around", "recall", "known_map"}
MAX_ACTIONS = 50
MAP_FILE = "warehouse_map.json"
COLOR_WORDS = ("red", "blue", "yellow", "purple", "orange", "green", "white", "black")

MOBILE_DOC = """
robot.places() -> list of known place names, e.g. ["station A", "station B", "packing station"]
robot.go_to(place) -> True/False   (drive to a known place and face its table)
robot.where_am_i() -> name of the place the robot is at, or None
robot.look_around() -> list of cubes seen on the table in front, e.g. ["red cube", "blue cube"]; also updates the robot's MAP
robot.recall(item) -> the place where the MAP last saw this item, or None
robot.known_map() -> the whole MAP as a dict {item: place}
robot.see(phrase) -> list of [x, y, z] positions seen by the front camera, best first; [] if not seen
robot.pick(phrase) -> True/False   (pick the named object from the table in front of the robot)
robot.place_on(phrase) -> True/False   (put the held object on / into the named thing in front of the robot)
robot.put_aside() -> True/False   (put the held object on a free spot of the table in front of the robot)
robot.is_holding() -> True/False   (the robot can carry only ONE object at a time)
robot.ask(question) -> short answer from looking through the front camera, e.g. robot.ask("Which object is on top of the red cube?") -> "blue cube"; returns "" if not sure
Phrases are short noun phrases, e.g. "red cube", "tray".
Driving takes time: check the MAP with recall() before exploring. The MAP can be out of date (people move things), so verify with look_around() when you arrive.
"""


class MobileAPI:
    def __init__(self, env):
        self.env = env
        self.log = []
        self.map = json.load(open(MAP_FILE)) if os.path.exists(MAP_FILE) else {}
        self.seen_here = {}   # place -> cubes seen by the last look_around there

    def _record(self, action, arg, result):
        self.log.append({"action": action, "arg": arg, "result": result})
        if len(self.log) > MAX_ACTIONS:
            raise RuntimeError("too many actions")

    def _save_map(self):
        json.dump(self.map, open(MAP_FILE, "w"), indent=2)

    def places(self):
        return list(self.env.locations)

    def go_to(self, place):
        ok, msg = self.env.go_to(str(place))
        self._record("go_to", place, msg)
        return ok

    def where_am_i(self):
        return self.env.at

    def look_around(self):
        from llm import spark_vision
        rgb, _ = self.env.camera("front")
        answer = spark_vision("List every cube you can see on the table in front of you, by color, "
                              "separated by commas, like 'red cube, blue cube'. Ignore trays. "
                              "If there are none, answer 'none'.", rgb)
        items = [s.strip(" .") for s in re.split(r",| and ", answer) if "cube" in s]
        if self.env.at:
            for it in items:
                self.map[it] = self.env.at
            for it, place in list(self.map.items()):   # forget items no longer seen here
                if place == self.env.at and it not in items:
                    self.map.pop(it)
            self.seen_here[self.env.at] = items
            self._save_map()
        self._record("look_around", self.env.at, items)
        return items

    def recall(self, item):
        return self.map.get(item)

    def known_map(self):
        return dict(self.map)

    def see(self, phrase):
        found = [[round(c, 3) for c in f["pos"]] for f in self.env.perceive(phrase)]
        if found and self.env.at:
            self.map[phrase] = self.env.at
            self._save_map()
        self._record("see", phrase, found)
        return found

    def pick(self, phrase):
        phrase = str(phrase).strip()
        if not any(c in phrase.lower() for c in COLOR_WORDS):
            self._record("pick", phrase, "REFUSED: not a valid object name")
            return False
        ok, msg = self.env.pick_seen(phrase)
        if ok:
            self.map.pop(phrase, None)   # now carried by the robot
            self._save_map()
        elif self.env.at in self.seen_here:
            msg += f"; last look_around here saw: {self.seen_here[self.env.at]}"
        self._record("pick", phrase, msg)
        return ok

    def place_on(self, phrase):
        ok, msg = self.env.place_on(phrase)
        self._record("place_on", phrase, msg)
        return ok

    def put_aside(self):
        ok, msg = self.env.put_aside()
        self._record("put_aside", "", msg)
        return ok

    def is_holding(self):
        return self.env.held is not None

    def ask(self, question):
        from llm import spark_vision
        rgb, _ = self.env.camera("front")
        answer = spark_vision(question, rgb)
        if not any(c in answer for c in COLOR_WORDS):   # "unknown", "gray block", "none"...
            answer = ""
        self._record("ask", question, answer)
        return answer