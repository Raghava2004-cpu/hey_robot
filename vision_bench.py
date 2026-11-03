"""Sim-to-real test, like a Roboflow test set: would the robots' vision still work on real-looking pictures?

For each table, the robot's front camera takes a picture in the simulator. Muse Image turns it into
3 new pictures: a realistic photo, the same photo in dim light, and a photo with clutter on the floor.
Then, on every picture:
  - SAM 3.1 must find every item on the table,
  - Muse Spark must name each found item correctly,
  - the box must stay at the same spot ON THE TABLE as in the simulator picture (IoU >= 0.5, measured
    relative to the table's box, so a slightly zoomed picture is fair), so the grasp point would be the same.

python vision_bench.py                    # 3 tables x 4 pictures (about 5-10 min, needs the hotspot)
python vision_bench.py --stations A C     # only some tables
Output: runs/bench/<table>_<style>.png (labelled), runs/bench/grid.png, runs/bench/vision_bench.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

STYLES = {
    "simulator": None,
    "photo": ("Turn this into a realistic photograph taken by a robot's camera in a real warehouse. Keep the "
              "camera angle, the table and every object exactly where they are, with the same size and shape."),
    "dim light": ("The same scene as a realistic photograph, but in dim evening light with soft shadows. Keep the "
                  "camera angle and every object exactly where it is."),
    "clutter": ("The same scene as a realistic photograph, with a few extra unrelated things on the FLOOR around "
                "the table (a cardboard box, a coffee cup). Keep the camera angle and every object on the table "
                "exactly where it is."),
}
FRAME = " Do not move, zoom or crop the camera: the picture frame must stay exactly the same."
OUT = "runs/bench"


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def table_box(img):
    """The table's box: item positions are compared RELATIVE to the table, so a picture that Muse Image
    zoomed or shifted a little is not counted as the items moving."""
    from perception import sam_find
    for ph in ("table", "dark grey table", "black table"):
        boxes = sam_find(img, ph)
        if boxes:
            return max(boxes, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]))["box"]
    return None


def on_table(box, tb):
    if box is None or tb is None:
        return box
    w, h = max(1.0, tb[2] - tb[0]), max(1.0, tb[3] - tb[1])
    return [(box[0] - tb[0]) / w, (box[1] - tb[1]) / h, (box[2] - tb[0]) / w, (box[3] - tb[1]) / h]


def find(img, item):
    from mobile_env import ALIASES
    from perception import sam_find
    for ph in ALIASES.get(item, [item]):
        boxes = sam_find(img, ph)
        if boxes:
            return max(boxes, key=lambda b: b.get("conf") or 0)
    return None


def main():
    import muse_eye
    from mobile_env import W, H
    from store_tasks import STORE_LAYOUT, store_scene
    args = sys.argv[1:]
    stations = [f"station {a}" for a in args[args.index("--stations") + 1:]] if "--stations" in args \
        else ["station A", "station B", "station C"]
    os.makedirs(OUT, exist_ok=True)
    env = store_scene()(gui=False)
    rows, tiles = [], []
    for st in stations:
        items = [n for n, (s, _) in STORE_LAYOUT.items() if s == st]
        env.go_to(st)
        env.clear_view()
        sim, _ = env.camera("front")
        ref = {it: find(sim, it) for it in items}                      # where SAM sees it in the simulator
        ref_table = table_box(sim)
        for style, prompt in STYLES.items():
            if prompt is None:
                img = sim
            else:
                try:
                    from imagine import imagine
                    out = np.asarray(imagine(prompt + FRAME, sim))[:, :, :3]
                    img = np.asarray(Image.fromarray(out.astype(np.uint8)).resize((W, H)))
                except Exception as e:
                    print(f"   {st} {style}: Muse Image failed ({str(e)[:80]})")
                    continue
            dets = []
            tb = table_box(img) if ref_table is not None else None
            for it in items:
                b = find(img, it)
                r = {"table": st, "style": style, "item": it, "found": b is not None,
                     "conf": round(float(b.get("conf") or 0), 3) if b else 0.0}
                if b:
                    ok, ans = muse_eye.confirm(img, b["box"], it)
                    rb = ref[it]["box"] if ref[it] else None
                    raw = iou(b["box"], rb)
                    aligned = iou(on_table(b["box"], tb), on_table(rb, ref_table)) if tb is not None else raw
                    r.update(spark=ans, named_right=bool(ok), iou_raw=round(raw, 2), iou=round(max(raw, aligned), 2))
                    dets.append(dict(b, label=f"{it} | Spark: {ans}"))
                else:
                    r.update(spark=None, named_right=False, iou=0.0)
                rows.append(r)
                print(f"   {st:10s} {style:10s} {it:13s} found={r['found']!s:5s} spark={r['spark']}  "
                      f"same spot on the table IoU={r['iou']} (whole picture {r.get('iou_raw', 0.0)})")
            pic = muse_eye.annotate(img, dets, f"{st} | {style} | SAM 3.1 + Muse Spark", chosen=-1)
            pic.save(os.path.join(OUT, f"{st.replace(' ', '_')}_{style.replace(' ', '_')}.png"))
            tiles.append(pic.resize((W // 2, H // 2)))
    env.close()
    # summary per style
    print("\n=========== SIM-TO-REAL SUMMARY ===========")
    summary = {}
    for style in STYLES:
        rs = [r for r in rows if r["style"] == style]
        if not rs:
            continue
        n = len(rs)
        found = sum(r["found"] for r in rs)
        named = sum(r["named_right"] for r in rs)
        same = sum(r["iou"] >= 0.5 for r in rs)
        summary[style] = {"items": n, "found_by_sam": found, "named_right_by_spark": named, "same_spot": same}
        print(f"{style:10s}  SAM found {found}/{n}   Spark named right {named}/{n}   same spot {same}/{n}")
    json.dump({"rows": rows, "summary": summary}, open(os.path.join(OUT, "vision_bench.json"), "w"), indent=2)
    if tiles:
        cols = len([s for s in STYLES])
        grid = Image.new("RGB", (cols * (W // 2), ((len(tiles) + cols - 1) // cols) * (H // 2)), (10, 10, 10))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * (W // 2), (i // cols) * (H // 2)))
        grid.save(os.path.join(OUT, "grid.png"))
        print(f"\nPictures: {OUT}/  (grid.png = everything in one picture)")


if __name__ == "__main__":
    main()