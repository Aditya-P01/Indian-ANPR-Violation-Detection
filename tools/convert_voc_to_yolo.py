"""Convert a Pascal-VOC plate dataset (images/ + annotations/*.xml) to YOLO format.

    python tools/convert_voc_to_yolo.py --src path/to/kaggle_dataset --dst datasets/plates

Creates  dst/images/{train,val}, dst/labels/{train,val} and dst/data.yaml.
All object classes are mapped to a single class: "license_plate".
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET


def convert(xml_path: str):
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename")
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    lines = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return fname, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with images/ and annotations/")
    ap.add_argument("--dst", default="datasets/plates")
    ap.add_argument("--val", type=float, default=0.2, help="validation fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_dir = os.path.join(args.src, "annotations")
    img_dir = os.path.join(args.src, "images")
    xmls = sorted(f for f in os.listdir(ann_dir) if f.endswith(".xml"))
    random.Random(args.seed).shuffle(xmls)
    n_val = int(len(xmls) * args.val)

    count = {"train": 0, "val": 0}
    for i, x in enumerate(xmls):
        split = "val" if i < n_val else "train"
        fname, lines = convert(os.path.join(ann_dir, x))
        img = os.path.join(img_dir, fname or "")
        if not fname or not os.path.exists(img):
            stem = os.path.splitext(x)[0]
            cands = [f for f in os.listdir(img_dir) if os.path.splitext(f)[0] == stem]
            if not cands:
                continue
            fname, img = cands[0], os.path.join(img_dir, cands[0])
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, sub, split), exist_ok=True)
        shutil.copy(img, os.path.join(args.dst, "images", split, fname))
        with open(os.path.join(args.dst, "labels", split, os.path.splitext(fname)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines))
        count[split] += 1

    with open(os.path.join(args.dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.dst)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: license_plate\n")
    print(f"train={count['train']}  val={count['val']}  -> {os.path.join(args.dst, 'data.yaml')}")


if __name__ == "__main__":
    main()
