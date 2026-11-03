import json
import os

from audit_tasks import AUDIT_DOC, AUDIT_METHODS, AUDIT_TEXT, AuditAPI, audit_check, audit_scene
from learner import learn
from team_skills import MAP_FILE

if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)

LIB = "audit_library.json"
lib = []
for f in ["mission_library.json", "landing_library.json"]:   # lessons from earlier missions
    if os.path.exists(f):
        lib += json.load(open(f))[-1:]
json.dump(lib, open(LIB, "w"), indent=2)
print(f"Audit memory starts with {len(lib)} lessons from earlier missions.")

r = learn(AUDIT_TEXT, audit_scene(), audit_check, library=LIB, api_cls=AuditAPI,
          api_doc=AUDIT_DOC, methods=AUDIT_METHODS, max_attempts=3, use_imagination=False)

print("\n=========== SUMMARY ===========")
print(f"Inventory audit: success={r['success']}  attempts={r['attempts']}  actions={r['actions']}")