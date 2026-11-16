"""Convert COCO-format plate datasets (Roboflow / Hugging Face exports) to YOLO format.

Expects split folders that each hold images + _annotations.coco.json, e.g.

    datasets/hf_raw/train/_annotations.coco.json
    datasets/hf_raw/valid/_annotations.coco.json
    datasets/hf_raw/test/_annotations.coco.json

    python tools/convert_coco_to_yolo.py --src datasets/hf_raw --dst datasets/plates

Creates dst/images/<split>, dst/labels/<split> and dst/data.yaml (one class: license_plate).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def convert_split(src_dir: str, dst: str, split: str) -> int:
    ann_path = os.path.join(src_dir, "_annotations.coco.json")
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    boxes = defaultdict(list)
    for a in coco["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    img_out = os.path.join(dst, "images", split)
    lbl_out = os.path.join(dst, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n = 0
    for im in coco["images"]:
        src = os.path.join(src_dir, im["file_name"])
        if not os.path.exists(src):
            continue
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for x, y, bw, bh in boxes.get(im["id"], []):
            if bw <= 1 or bh <= 1:
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"0 {min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} {min(bw / w, 1):.6f} {min(bh / h, 1):.6f}")
        shutil.copy(src, os.path.join(img_out, im["file_name"]))
        with open(os.path.join(lbl_out, os.path.splitext(im["file_name"])[0] + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/ valid/ test/ COCO splits")
    ap.add_argument("--dst", default="datasets/plates")
    args = ap.parse_args()

    found = {}
    for name in sorted(os.listdir(args.src)):
        d = os.path.join(args.src, name)
        if name in SPLITS and os.path.isfile(os.path.join(d, "_annotations.coco.json")):
            found[SPLITS[name]] = convert_split(d, args.dst, SPLITS[name])
    if "train" not in found or "val" not in found:
        raise SystemExit(f"need at least train/ and valid/ splits in {args.src}, found {list(found)}")

    yaml_lines = [f"path: {os.path.abspath(args.dst)}", "train: images/train", "val: images/val"]
    if "test" in found:
        yaml_lines.append("test: images/test")
    yaml_lines += ["names:", "  0: license_plate", ""]
    with open(os.path.join(args.dst, "data.yaml"), "w") as f:
        f.write("\n".join(yaml_lines))
    print("  ".join(f"{k}={v}" for k, v in found.items()), "->", os.path.join(args.dst, "data.yaml"))


if __name__ == "__main__":
    main()
