import os
import shutil

from learner import learn
from mobile_skills import MAP_FILE, MOBILE_DOC, MOBILE_METHODS, MobileAPI
from warehouse_tasks import ORDERS, RESULT

if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)                          # the robot starts with NO map
LIB = "warehouse_library.json"
src = "mobile_library.json" if os.path.exists("mobile_library.json") else "skills_library.json"
shutil.copy(src, LIB)                            # but with lessons from earlier robots
kw = dict(library=LIB, api_cls=MobileAPI, api_doc=MOBILE_DOC, methods=MOBILE_METHODS,
          max_attempts=3)

results = {}
for key, title in [("order1", "ORDER 1: two items, unknown places, one hidden"),
                   ("order2", "ORDER 2: map known from order 1"),
                   ("order3", "ORDER 3: a worker MOVED the blue cube (map is wrong)")]:
    print(f"\n=========== {title} ===========")
    r = learn(*ORDERS[key], **kw)
    results[key] = (r, RESULT["meters"])

print("\n=========== SUMMARY ===========")
for key, (r, meters) in results.items():
    print(f"{key}: success={r['success']}  attempts={r['attempts']}  actions={r['actions']}  "
          f"meters driven (last attempt)={meters}")