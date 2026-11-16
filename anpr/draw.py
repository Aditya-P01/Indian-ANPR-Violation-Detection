"""Drawing helpers for the annotated demo video."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .geometry import Line
from .signal import GREEN, RED
from .violations import Track

C_OK = (60, 200, 60)
C_BAD = (40, 40, 230)
C_LINE = (0, 215, 255)
C_STOP = (60, 60, 255)
C_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img: np.ndarray, text: str, org: Tuple[int, int], color, scale: float = 0.55, thick: int = 1) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, C_TEXT, thick, cv2.LINE_AA)


def draw_lines(img: np.ndarray, lines: Iterable[Line], signal_state: str) -> None:
    for ln in lines:
        p1, p2 = ln.as_int()
        if ln.name == "STOP LINE":
            color = C_STOP if signal_state == RED else C_OK
            cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
        else:
            cv2.line(img, p1, p2, C_LINE, 2, cv2.LINE_AA)
        label(img, ln.name, (p1[0] + 4, p1[1] - 6), (40, 40, 40), 0.5)


def draw_track(img: np.ndarray, tr: Track, show_plate_box: bool = True) -> None:
    x1, y1, x2, y2 = tr.box
    bad = bool(tr.violations)
    color = C_BAD if bad else C_OK
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if bad else 2)
    parts = [f"#{tr.track_id} {tr.cls_name}"]
    if tr.plate:
        parts.append(tr.plate)
    if tr.speed_kmph is not None:
        parts.append(f"{tr.speed_kmph:.0f} km/h")
    label(img, " | ".join(parts), (x1, y1 - 4), color)
    if bad:
        label(img, " ".join(tr.violations), (x1, y2 + 22), C_BAD, 0.5)
    if show_plate_box and tr.plate_box is not None:
        px1, py1, px2, py2 = tr.plate_box
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 200, 0), 2)


def draw_hud(img: np.ndarray, signal_state: str, counts: dict, n_vehicles: int, fps: float,
             alerts: List[str], plate_mode: str) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (330, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    lamp = (40, 40, 230) if signal_state == RED else ((60, 200, 60) if signal_state == GREEN else (150, 150, 150))
    cv2.circle(img, (25, 25), 12, lamp, -1)
    cv2.putText(img, f"SIGNAL: {signal_state}", (45, 32), FONT, 0.65, C_TEXT, 2, cv2.LINE_AA)
    y = 60
    for k, v in counts.items():
        cv2.putText(img, f"{k.replace('_', ' ')}: {v}", (12, y), FONT, 0.55, C_TEXT, 1, cv2.LINE_AA)
        y += 22
    cv2.putText(img, f"Vehicles: {n_vehicles}   FPS: {fps:.1f}", (12, y), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Plate: {plate_mode}", (12, y + 20), FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    for i, a in enumerate(alerts[-3:]):
        (tw, _), _ = cv2.getTextSize(a, FONT, 0.6, 2)
        label(img, a, (w - tw - 20, 30 + i * 30), C_BAD, 0.6, 2)
