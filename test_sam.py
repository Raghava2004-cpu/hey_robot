from perception import sam_find, draw_boxes
from robot_env import RobotEnv

CUBE_HALF = 0.025  # camera sees the top face

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)
rgb, depth = env.camera()

results = {}
for name in ["red", "blue"]:
    phrase = f"{name} cube"
    boxes = sam_find(rgb, phrase)
    results[phrase] = boxes
    print(f"\n'{phrase}': {len(boxes)} match(es)")
    true = env.get_pos(name)
    for b in boxes:
        u, v = b["center"]
        est = env.pixel_to_world(u, v, depth)
        est[2] -= CUBE_HALF
        err = sum((e - t) ** 2 for e, t in zip(est, true)) ** 0.5
        print(f"  box={b['box']} conf={b['conf']}  3D error={err*100:.1f} cm")

draw_boxes(rgb, results, "sam_view.png")
print("\nSaved sam_view.png")