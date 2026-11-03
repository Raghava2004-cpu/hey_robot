import json
import os

from learner import learn
from mission_tasks import MISSIONS
from team_skills import MAP_FILE, TEAM_DOC, TEAM_METHODS, TeamAPI

# the robots start with NO map of the warehouse
if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)

# but with lessons learned earlier by the warehouse robot and the drone
LIB = "mission_library.json"
lib = []
for f in ["warehouse_library.json", "landing_library.json"]:
    if os.path.exists(f):
        lib += json.load(open(f))[-1:]
json.dump(lib, open(LIB, "w"), indent=2)
print(f"Mission memory starts with {len(lib)} lessons from earlier robots.")

r = learn(*MISSIONS["mission1"], library=LIB, api_cls=TeamAPI, api_doc=TEAM_DOC,
          methods=TEAM_METHODS, max_attempts=3,
          imagine_text="The banana and the soup can are inside the green tray on the packing station table.")

print("\n=========== SUMMARY ===========")
print(f"Full mission: success={r['success']}  attempts={r['attempts']}  actions={r['actions']}")