"""Hey Robot's eyes on the REAL world: the same SAM 3.1 + Muse Spark pipeline the robots use, on real photos.

1) Phone photos -> a results table (real-camera evidence)
   Put photos in a folder named like  <items>__<light>.jpg   (items joined by "_", spaces as "-"):
       banana_scissors__normal.jpg     banana_scissors__dim.jpg     apple_soup-can__clutter.jpg
   python real_eye.py --photos real_photos
   For every photo: SAM 3.1 must find each listed item, Muse Spark must name each box correctly, and SAM
   is also asked for 2 things that are NOT in the photo (a false alarm counts against it).
   Output: runs/real/*.png (labelled), runs/real/grid.png, runs/real/real_eye.json

(The live webcam is used in the command center: python command_center.py --loop --home --webcam)
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

OUT = "runs/real"
REAL_CATALOG = ["banana", "apple", "orange", "pear", "strawberry", "grapes", "lemon", "tomato", "scissors",
                "soup can", "can", "bottle", "cup", "mug", "phone", "pen", "book", "box", "remote control",
                "keys", "glasses", "headphones", "spoon", "fork", "knife", "plate", "bowl", "tape", "clamp"]
DECOYS = ["banana", "scissors", "apple", "bottle", "mug", "phone", "book", "spoon"]


def load(path, max_w=1024):
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    return np.array(img)


def find_all(rgb, names):
    """SAM 3.1 for several names in parallel -> {name: [boxes]}."""
    from perception import sam_find
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(names)))) as pool:
        return dict(zip(names, pool.map(lambda n: sam_find(rgb, n), names)))


def name_crop(rgb, box, catalog):
    """Muse Spark names the object in the box (crop with a margin)."""
    from llm import spark_vision
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = rgb.shape[:2]
    m = max(12, int(0.2 * max(x1 - x0, y1 - y0)))
    crop = rgb[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    q = ("This is a crop from a real camera photo. Which object is in the middle of it? "
         f"Choose exactly one: {', '.join(catalog)}, other. Reply with the name only.")
    try:
        return str(spark_vision(q, crop)).strip().lower().strip("*\"'. ")
    except Exception as e:
        return f"error {str(e)[:30]}"


def same_thing(answer, item):
    a, i = (answer or "").lower().strip(), item.lower()
    if not a or a.startswith("error"):
        return False
    return i in a or (len(a) > 2 and a in i) or (i == "soup can" and "can" in a)


# ---------------------------------------------------------------- 1) phone photos
def run_photos(folder):
    import muse_eye
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print(f"No photos in {folder}. Name them like banana_scissors__dim.jpg")
        return
    rows, tiles = [], []
    for f in files:
        stem = os.path.splitext(f)[0]
        left, _, light = stem.partition("__")
        items = [w.replace("-", " ") for w in left.split("_") if w]
        light = light or "normal"
        decoys = [d for d in DECOYS if all(not same_thing(d, it) for it in items)][:2]
        rgb = load(os.path.join(folder, f))
        found = find_all(rgb, items + decoys)
        catalog = sorted(set(REAL_CATALOG + items))
        dets = []
        for it in items:
            boxes = sorted(found[it], key=lambda b: -(b.get("conf") or 0))
            r = {"photo": f, "light": light, "item": it, "found": bool(boxes)}
            if boxes:
                ans = name_crop(rgb, boxes[0]["box"], catalog)
                r.update(conf=round(float(boxes[0].get("conf") or 0), 3), spark=ans, named_right=same_thing(ans, it))
                dets.append(dict(boxes[0], label=f"{it} | Spark: {ans}"))
            else:
                r.update(conf=0.0, spark=None, named_right=False)
            rows.append(r)
            print(f"   {f:38s} {it:12s} found={r['found']!s:5s} spark={r['spark']}")
        for d in decoys:
            fa = bool(found[d])
            rows.append({"photo": f, "light": light, "decoy": d, "false_alarm": fa})
            print(f"   {f:38s} (not in photo) {d:10s} SAM false alarm={fa}")
        pic = muse_eye.annotate(rgb, dets, f"REAL PHOTO | {light} | SAM 3.1 + Muse Spark", chosen=-1)
        pic.save(os.path.join(OUT, f"{stem}.png"))
        tiles.append(pic.resize((480, int(480 * pic.height / pic.width))))
    print("\n=========== REAL-CAMERA SUMMARY ===========")
    summary = {}
    for light in sorted({r["light"] for r in rows}):
        it = [r for r in rows if r["light"] == light and "item" in r]
        dc = [r for r in rows if r["light"] == light and "decoy" in r]
        summary[light] = {"items": len(it), "found": sum(r["found"] for r in it),
                          "named_right": sum(r["named_right"] for r in it),
                          "decoys": len(dc), "false_alarms": sum(r["false_alarm"] for r in dc)}
        s = summary[light]
        print(f"{light:8s}  SAM found {s['found']}/{s['items']}   Spark named right {s['named_right']}/{s['items']}"
              f"   SAM false alarms {s['false_alarms']}/{s['decoys']}")
    json.dump({"rows": rows, "summary": summary}, open(os.path.join(OUT, "real_eye.json"), "w"), indent=2)
    if tiles:
        cols = min(3, len(tiles))
        th = max(t.height for t in tiles)
        grid = Image.new("RGB", (cols * 480, ((len(tiles) + cols - 1) // cols) * th), (10, 10, 10))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * 480, (i // cols) * th))
        grid.save(os.path.join(OUT, "grid.png"))
        print(f"\nPictures: {OUT}/  (grid.png = all photos)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--photos" in a:
        run_photos(a[a.index("--photos") + 1])
    else:
        print(__doc__)
