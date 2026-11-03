import pybullet as p
from PIL import Image

from cube_skills import CubeAPI
from cube_tasks import cube_scene, spell_check, stack_check, tower_height


def snap(env, path):
    rgb, _ = env.camera("front")
    Image.fromarray(rgb).save(path)


def where(env):
    return {env.letter_of[n]: [round(v, 3) for v in env.get_pos(n)] for n in env.letter_of}


# ---- Part 1: see + read + stack ----
env = cube_scene(seed=1)(gui=True)
api = CubeAPI(env)
seen = api.look()
print("SEEN :", [(b["letter"], b["pos"]) for b in seen])
if all(b["letter"] == "?" for b in seen):
    print("SPARK REPLY:", api.last_reply[:400])
x, y = api.build_spot()
z = api.table_height()
for b in seen[:4]:
    if api.pick(*b["pos"]):
        api.place(x, y, z)
        z += 0.05
        print(f"  tower height = {tower_height(env)}   blocks now: {where(env)}")
print("STACK CHECK:", stack_check(4)(env))
snap(env, "runs/stack_after.png")
env.close()

# ---- Part 2: read letters + spell META ----
env = cube_scene(seed=2)(gui=True)
api = CubeAPI(env)
seen = api.look()
print("SEEN :", [(b["letter"], b["pos"]) for b in seen])
if all(b["letter"] == "?" for b in seen):
    print("SPARK REPLY:", api.last_reply[:400])
for (sx, sy), L in zip(api.word_slots(4), "META"):
    b = next((b for b in seen if b["letter"] == L), None)
    if b and api.pick(*b["pos"]):
        api.place(sx, sy, api.table_height())
print("SPELL CHECK:", spell_check("META")(env))
snap(env, "runs/spell_after.png")
while p.isConnected():
    env.step()