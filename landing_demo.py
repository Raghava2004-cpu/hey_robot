import os

from drone_skills import DRONE_DOC, DRONE_METHODS, DroneAPI
from landing_tasks import LANDING_TASKS
from learner import learn

LIB = "landing_library.json"
if os.path.exists(LIB):
    os.remove(LIB)
kw = dict(library=LIB, api_cls=DroneAPI, api_doc=DRONE_DOC, methods=DRONE_METHODS, max_attempts=4)

print("=========== LANDING A: unknown wind, moving robot (no experience) ===========")
a = learn(*LANDING_TASKS["land_A"], **kw)

print("\n=========== LANDING B: faster robot, new start (with memory) ===========")
b = learn(*LANDING_TASKS["land_B"], **kw)

print("\n=========== SUMMARY ===========")
print(f"Landing A: success={a['success']}  attempts={a['attempts']}")
print(f"Landing B: success={b['success']}  attempts={b['attempts']}")