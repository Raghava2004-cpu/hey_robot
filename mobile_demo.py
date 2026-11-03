import os
import shutil

from learner import learn
from mobile_skills import MOBILE_DOC, MOBILE_METHODS, MobileAPI
from mobile_tasks import MOBILE_TASKS

LIB = "mobile_library.json"
if os.path.exists("skills_library.json"):
    shutil.copy("skills_library.json", LIB)   # start with what the FIXED arm learned
print("Starting memory copied from the fixed-arm robot.")

kw = dict(library=LIB, api_cls=MobileAPI, api_doc=MOBILE_DOC, methods=MOBILE_METHODS)

print("\n=========== MOBILE 1: simple fetch ===========")
a = learn(*MOBILE_TASKS["fetch"], **kw)

print("\n=========== MOBILE 2: red cube HIDDEN under blue ===========")
b = learn(*MOBILE_TASKS["fetch_hidden"], **kw)

print("\n=========== SUMMARY ===========")
print(f"Simple fetch: success={a['success']}  attempts={a['attempts']}  actions={a['actions']}")
print(f"Hidden fetch: success={b['success']}  attempts={b['attempts']}  actions={b['actions']}")