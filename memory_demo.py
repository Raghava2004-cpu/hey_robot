import os

from learner import LIBRARY, learn
from tasks import TASKS

if os.path.exists(LIBRARY):
    os.remove(LIBRARY)  # start with an empty memory

print("=========== TASK A: blue hidden under red (empty memory) ===========")
a = learn(*TASKS["blue_hidden"])

print("\n=========== TASK B: NEW task, purple hidden under yellow (with memory) ===========")
b = learn(*TASKS["purple_hidden"])

print("\n=========== SUMMARY ===========")
print(f"Task A (no memory):   success={a['success']}  attempts={a['attempts']}  actions={a['actions']}")
print(f"Task B (with memory): success={b['success']}  attempts={b['attempts']}  actions={b['actions']}")