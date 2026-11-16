"""Click the stop line / speed lines on a frame of your video and get config values.

    python tools/pick_points.py --video data/traffic.mp4 [--frame 0] [--width 1280]

Click 2 points for each line in this order:
    1) STOP LINE   2) line A (speed entry)   3) line B (speed exit)
optionally then 2 corners of the traffic-light ROI (top-left, bottom-right).
Press  u = undo,  s / Enter = print config,  q = quit.
"""

from __future__ import annotations

import argparse

import cv2

NAMES = ["stop_line", "line_a", "line_b", "light_roi"]
COLORS = [(60, 60, 255), (0, 215, 255), (0, 215, 255), (255, 200, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    if args.width and frame.shape[1] != args.width:
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * args.width / frame.shape[1])))
    H, W = frame.shape[:2]
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
            pts.append((x, y))

    win = "pick points (u=undo, s=print, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    def dump():
        print("\n# ---- paste into config.yaml (rules section) ----")
        for i in range(0, len(pts) - 1, 2):
            name = NAMES[i // 2]
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            n = [[round(x1 / W, 3), round(y1 / H, 3)], [round(x2 / W, 3), round(y2 / H, 3)]]
            if name == "light_roi":
                print(f"light_roi: [{n[0][0]}, {n[0][1]}, {n[1][0]}, {n[1][1]}]")
            else:
                print(f"{name}: {n}")

    while True:
        img = frame.copy()
        for i, p in enumerate(pts):
            c = COLORS[i // 2]
            cv2.circle(img, p, 5, c, -1)
            if i % 2 == 1:
                if i // 2 == 3:
                    cv2.rectangle(img, pts[i - 1], p, c, 2)
                else:
                    cv2.line(img, pts[i - 1], p, c, 2)
                cv2.putText(img, NAMES[i // 2], (pts[i - 1][0], pts[i - 1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        nxt = NAMES[min(len(pts) // 2, 3)] if len(pts) < 8 else "done"
        cv2.putText(img, f"next: {nxt}", (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (ord("s"), 13):
            dump()
        elif k == ord("q"):
            break
    dump()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
