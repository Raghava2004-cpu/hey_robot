import os
import sys

from agent_policy import act_live
from cube_skills import CUBE_DOC, CUBE_METHODS, CubeAPI
from cube_tasks import STACK_TASK, cube_scene, spell_check, spell_task, stack_check
from learner import learn

LIB = "cubes_library.json"
API = dict(api_cls=CubeAPI, api_doc=CUBE_DOC, methods=CUBE_METHODS)
STACK_GOAL = ("A single tower of 4 letter blocks stacked straight on top of each other, standing on the "
              "far side of the table (farthest from the robot); the other blocks lie near the robot.")
SPELL_GOAL = ("Four letter blocks standing in one straight row on the far side of the table (farthest from "
              "the robot), spelling META from left to right; the other blocks lie near the robot.")

if len(sys.argv) > 1 and sys.argv[1] == "live":
    print("=========== LIVE AGENT: Spark picks one action at a time ===========")
    r = act_live(spell_task("META"), cube_scene(seed=7), spell_check("META"), **API)
    print(f"\nLIVE: success={r['success']}  steps={r['steps']}  model calls={r['model_calls']}  "
          f"time={r['seconds']} s\nCHECK: {r['check']}")
else:
    if os.path.exists(LIB):
        os.remove(LIB)
    print("=========== TASK 1: stack a tower (empty library) ===========")
    a = learn(STACK_TASK, cube_scene(seed=101), stack_check(4), library=LIB, imagine_text=STACK_GOAL, **API)
    print("\n=========== TASK 2: spell META (library from task 1) ===========")
    b = learn(spell_task("META"), cube_scene(seed=7), spell_check("META"), library=LIB,
              imagine_text=SPELL_GOAL, **API)
    print("\n=========== SUMMARY ===========")
    print(f"Stack: success={a['success']}  attempts={a['attempts']}")
    print(f"Spell: success={b['success']}  attempts={b['attempts']}")