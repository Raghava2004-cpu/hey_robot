"""Ask Muse Spark a question from PowerShell (optionally with files and pictures attached).

python ask_spark.py "your question"
python ask_spark.py --file README.md "your question"
python ask_spark.py --file README.md --file muse_eye.py "your question"
python ask_spark.py --image runs/bench/grid.png "your question"
python ask_spark.py --effort high --file README.md "your question"

Every answer is also saved in runs/spark_answers.md.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

from llm import spark
from perception import _to_data_url


def main():
    args = sys.argv[1:]
    files, images, effort = [], [], "medium"
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--file":
            files.append(args.pop(0))
        elif flag == "--image":
            images.append(args.pop(0))
        elif flag == "--effort":
            effort = args.pop(0)
    question = " ".join(args).strip()
    if not question:
        print(__doc__)
        return
    text = question
    for f in files:
        body = open(f, encoding="utf-8", errors="replace").read()[:60000]
        text += f"\n\n===== FILE: {f} =====\n{body}"
    content = [{"type": "text", "text": text}]
    for im in images:
        rgb = np.array(Image.open(im).convert("RGB"))
        content.append({"type": "image_url", "image_url": {"url": _to_data_url(rgb)}})
    messages = [{"role": "system", "content":
                 "You are a senior robotics and AI engineer and an experienced hackathon judge. "
                 "Give concrete, specific, honest answers. Use short sentences and numbered lists."},
                {"role": "user", "content": content if images else text}]
    print(f"Asking Muse Spark (effort={effort})...")
    answer, usage = spark(messages, effort=effort)
    print("\n" + answer)
    os.makedirs("runs", exist_ok=True)
    with open("runs/spark_answers.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n\n## {time.strftime('%Y-%m-%d %H:%M')}  Q: {question}\n"
                 f"(files: {files}, images: {images})\n\n{answer}\n")
    print("\n(saved in runs/spark_answers.md)")


if __name__ == "__main__":
    main()
