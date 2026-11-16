"""Make a short synthetic test clip (no download needed) to smoke-test the pipeline.

Uses the bus photo that ships with ultralytics, pastes a rendered Indian
plate on it, and slides it down the frame so it crosses all three lines.

    python tools/make_synthetic_video.py --out data/synthetic.mp4 --plate MH12AB1234
    python main.py --source data/synthetic.mp4
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def render_plate(text: str, w: int = 260, h: int = 60) -> np.ndarray:
    plate = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(plate, (2, 2), (w - 3, h - 3), (0, 0, 0), 3)
    scale = 1.4
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    scale *= min(1.0, (w - 20) / tw)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(plate, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    return plate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--plate", default="MH12AB1234")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    import ultralytics

    bus_path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    bus = cv2.imread(bus_path)
    if bus is None:
        raise SystemExit(f"could not read {bus_path}")
    # crop the bus (left part of the image) and add a plate on its front bumper
    bus = bus[220:900, 0:700].copy()
    plate = render_plate(args.plate, 230, 52)
    py, px = 430, 50   # lower front of the bus, inside its bounding box
    bus[py:py + plate.shape[0], px:px + plate.shape[1]] = plate

    W, H = 1280, 1440
    sprite = cv2.resize(bus, (560, 544))
    n = int(args.seconds * args.fps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    road = np.full((H, W, 3), 90, np.uint8)
    for x in range(W // 2 - 4, W // 2 + 4):
        road[::80, x] = 255
    for i in range(n):
        frame = road.copy()
        y = int(-200 + (H - 300) * i / (n - 1))   # bus moves top -> bottom
        x = (W - sprite.shape[1]) // 2
        y0, y1 = max(y, 0), min(y + sprite.shape[0], H)
        if y1 > y0:
            frame[y0:y1, x:x + sprite.shape[1]] = sprite[y0 - y:y1 - y]
        writer.write(frame)
    writer.release()
    print(f"wrote {args.out}  ({n} frames, plate {args.plate})")


if __name__ == "__main__":
    main()
