from PIL import Image

from perception import sam_find
from warehouse_tasks import BASE, warehouse

env = warehouse(BASE, hidden=("purple", "orange"))(gui=True)
print(env.go_to("station C"))
rgb, _ = env.camera("front")
Image.fromarray(rgb).save("station_c.png")
print("saved station_c.png")

print("true orange:", [round(v, 3) for v in env.get_pos("orange")])
print("true purple:", [round(v, 3) for v in env.get_pos("purple")])

for phrase in ["orange cube", "orange block", "orange box", "purple cube",
               "purple block", "cube", "stacked cubes"]:
    boxes = sam_find(rgb, phrase)
    print(f"{phrase:15s} -> {len(boxes)} match(es)  "
          f"conf={[round(b['conf'], 2) for b in boxes]}  boxes={[b['box'] for b in boxes]}")