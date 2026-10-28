from PIL import Image
from robot_env import RobotEnv

env = RobotEnv(gui=True)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)

rgb, depth = env.camera()
Image.fromarray(rgb).save("cam.png")
print("Saved cam.png")

for name in ["red", "blue"]:
    true = env.get_pos(name)
    u, v = env.world_to_pixel(true)
    est = env.pixel_to_world(u, v, depth)
    print(f"{name}: pixel=({u:.0f},{v:.0f})  "
          f"x,y error={abs(est[0]-true[0])*100:.1f}cm,{abs(est[1]-true[1])*100:.1f}cm  "
          f"z error={(est[2]-true[2])*100:.1f}cm")