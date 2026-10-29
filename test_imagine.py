import base64
import os

from openai import OpenAI
from PIL import Image

from robot_env import RobotEnv

client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

env = RobotEnv(gui=False)
env.add_cube("red", [0.55, 0.10], [1, 0, 0, 1])
env.add_cube("blue", [0.65, 0.15], [0, 0, 1, 1])
env.step(120)
rgb, _ = env.camera()
Image.fromarray(rgb).save("real_before.png")

with open("real_before.png", "rb") as f:
    resp = client.images.edit(
        model="muse-image-1.0",
        prompt=("This is a top-down robot camera view of a table. Move ONLY the red cube "
                "into the center of the green tray. Keep everything else exactly the same: "
                "same camera angle, same blue cube position, same robot arm, same lighting."),
        image=f,
        n=1,
    )

item = resp.data[0]
if getattr(item, "b64_json", None):
    open("imagined_after.png", "wb").write(base64.b64decode(item.b64_json))
    print("Saved imagined_after.png")
else:
    print("Image URL (open it in a browser):", item.url)