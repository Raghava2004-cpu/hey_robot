import os

from learner import learn
from tasks import THROW_TASKS

LIB = "throw_library.json"
if os.path.exists(LIB):
    os.remove(LIB)  # start with an empty physics memory

print("=========== THROW A: tray at 1.05 m (no experience) ===========")
a = learn(*THROW_TASKS["throw_A"], library=LIB, max_attempts=4)

print("\n=========== THROW B: tray moved to a NEW spot (with memory) ===========")
b = learn(*THROW_TASKS["throw_B"], library=LIB, max_attempts=4)

print("\n=========== SUMMARY ===========")
print(f"Throw A (no memory):   success={a['success']}  attempts={a['attempts']}")
print(f"Throw B (with memory): success={b['success']}  attempts={b['attempts']}")