import json
import os
import sys

from audit_tasks import AUDIT_DOC, AUDIT_METHODS, AUDIT_TEXT, AuditAPI, audit_check, audit_scene
from learner import learn
from mission_tasks import mission_check, mission_scene
from team_skills import MAP_FILE, TEAM_DOC, TEAM_METHODS, TeamAPI
from voice import listen

# words the transcript may contain -> item names the robots know
KEYWORDS = {"banana": "banana", "soup": "soup can", "mustard": "mustard bottle",
            "brick": "foam brick", "gelatin": "gelatin box", "jello": "gelatin box"}
CONTEXT = (" The drone can scout the stations; the ground robot carries one item at a time and puts "
           "it in the tray at the packing station. When the order is done, the ground robot drives "
           "away and the drone must land on the landing pad on top of the moving ground robot. "
           "Wind may be blowing.")


def get_order():
    args = sys.argv[1:]
    if args and args[0] == "--text":            # typed fallback, e.g. no microphone
        return " ".join(args[1:])
    seconds = int(args[1]) if len(args) > 1 and args[0] == "--seconds" else 6
    return listen(seconds)


def make_library():
    lib = []
    for f in ["mission_library.json", "landing_library.json"]:   # lessons from earlier missions
        if os.path.exists(f):
            lib += json.load(open(f))[-1:]
    json.dump(lib, open("voice_library.json", "w"), indent=2)
    return "voice_library.json"


order = get_order().strip()
print(f'🗣️  HEARD: "{order}"')
if not order:
    sys.exit('Nothing was heard. Try again, or use: python voice_order.py --text "your order"')
low = order.lower()
if os.path.exists(MAP_FILE):
    os.remove(MAP_FILE)
lib = make_library()

if "audit" in low or "inventory" in low:
    task = f'Spoken order from the warehouse operator: "{order}". ' + AUDIT_TEXT
    r = learn(task, audit_scene(), audit_check, library=lib, api_cls=AuditAPI,
              api_doc=AUDIT_DOC, methods=AUDIT_METHODS, use_imagination=False)
else:
    targets = []
    for word, item in KEYWORDS.items():
        if word in low and item not in targets:
            targets.append(item)
    if not targets:
        sys.exit("I did not hear a known item (banana, soup can, mustard bottle, foam brick, gelatin box).")
    print("📦 ITEMS:", targets)
    task = f'Spoken order from the warehouse operator: "{order}".' + CONTEXT
    goal = f"The {' and the '.join(targets)} are inside the green tray on the packing station table."
    r = learn(task, mission_scene(), mission_check(targets), library=lib, api_cls=TeamAPI,
              api_doc=TEAM_DOC, methods=TEAM_METHODS, imagine_text=goal)

print("\n=========== SUMMARY ===========")
print(f'Order: "{order}"  ->  success={r["success"]}  attempts={r["attempts"]}  actions={r["actions"]}')