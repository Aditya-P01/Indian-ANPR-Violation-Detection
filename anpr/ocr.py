"""EasyOCR wrapper tuned for number plates (pre-processing + 1/2-line plates)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .plate_text import normalize, score

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class PlateRead:
    text: str                      # normalised plate text
    raw: str                       # raw OCR text
    conf: float                    # mean OCR confidence (0-1)
    valid: bool                    # matches the plate format
    score: float                   # voting weight
    box: Optional[Tuple[int, int, int, int]] = None  # plate box inside the vehicle crop
    crop: Optional[np.ndarray] = None                # plate image


def preprocess(plate: np.ndarray, target_h: int = 96) -> np.ndarray:
    """Grayscale, upscale and boost contrast so small plates are readable."""
    if plate.ndim == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return gray
    scale = target_h / float(h)
    if scale > 1.0:
        gray = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def deskew(gray: np.ndarray, max_angle: float = 25.0, min_angle: float = 1.5) -> Optional[np.ndarray]:
    """Straighten a tilted plate using its long, near-horizontal edges (plate border, text baseline).

    Returns the rotated image, or None when the plate is already level or no angle can be found.
    """
    h, w = gray.shape[:2]
    if w < 40 or h < 15:
        return None
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(20, w // 4),
                            minLineLength=w // 3, maxLineGap=max(5, w // 30))
    if lines is None:
        return None
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines, dtype=np.float64).reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in 5
        if x2 == x1:
            continue
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return None
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    angle = float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])  # length-weighted median
    if abs(angle) < min_angle:
        return None
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _merge_lines(results: Sequence) -> Tuple[str, float]:
    """Join EasyOCR boxes top-to-bottom, left-to-right (handles 2-line plates)."""
    items = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append((min(ys), max(ys), min(xs), text, float(conf)))
    if not items:
        return "", 0.0
    items.sort(key=lambda t: (t[0] + t[1]) / 2)
    rows: List[List[tuple]] = []
    for it in items:
        cy = (it[0] + it[1]) / 2
        if rows:
            last = rows[-1]
            top = min(r[0] for r in last)
            bot = max(r[1] for r in last)
            if top <= cy <= bot:
                last.append(it)
                continue
        rows.append([it])
    text = "".join("".join(r[3] for r in sorted(row, key=lambda r: r[2])) for row in rows)
    total_len = sum(max(len(it[3]), 1) for it in items)
    conf = sum(it[4] * max(len(it[3]), 1) for it in items) / total_len
    return text, conf


class PlateOCR:
    def __init__(self, languages=("en",), gpu: bool = False, fmt: str = "india", min_conf: float = 0.25):
        import easyocr  # imported lazily: heavy

        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        self.fmt = fmt
        self.min_conf = min_conf

    def read_plate(self, plate: np.ndarray) -> Optional[PlateRead]:
        """OCR an already-cropped plate image."""
        if plate is None or plate.size == 0:
            return None
        img = preprocess(plate)

        def variants():
            # plain crop first; deskew / binarise are only retries when no confident valid read yet
            yield img
            straight = deskew(img)
            if straight is not None:
                yield straight
            _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yield otsu

        best: Optional[PlateRead] = None
        for v in variants():
            results = self.reader.readtext(v, allowlist=ALNUM, detail=1, paragraph=False,
                                           text_threshold=0.5, low_text=0.3)
            raw, conf = _merge_lines(results)
            if not raw or conf < self.min_conf:
                continue
            text, valid = normalize(raw, self.fmt)
            if len(text) < 4:
                continue
            read = PlateRead(text, raw, conf, valid, score(text, conf, valid, self.fmt), crop=plate)
            if best is None or read.score > best.score:
                best = read
            if best.valid and best.conf >= 0.6:
                break
        return best

    def find_and_read(self, vehicle: np.ndarray) -> Optional[PlateRead]:
        """Fallback when no plate detector is available.

        Runs EasyOCR text detection on the vehicle crop and keeps the text
        region (or vertically stacked pair, for 2-line plates) that looks
        most like a number plate.
        """
        if vehicle is None or vehicle.size == 0:
            return None
        h, w = vehicle.shape[:2]
        scale = 1.0
        img = vehicle
        if w < 320:
            scale = 320.0 / w
            img = cv2.resize(vehicle, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        results = self.reader.readtext(img, allowlist=ALNUM, detail=1, paragraph=False)
        if not results:
            return None

        def box_of(group):
            xs = [p[0] for r in group for p in r[0]]
            ys = [p[1] for r in group for p in r[0]]
            return min(xs), min(ys), max(xs), max(ys)

        candidates = [[r] for r in results]
        # same-row neighbours ("MH 12" + "AB 1234") -> one plate
        rows = sorted(results, key=lambda r: box_of([r])[0])
        for i, a in enumerate(rows):
            group = [a]
            for b in rows[i + 1:]:
                gx1, gy1, gx2, gy2 = box_of(group)
                bx1, by1, bx2, by2 = box_of([b])
                v_overlap = min(gy2, by2) - max(gy1, by1)
                h = max(gy2 - gy1, by2 - by1)
                if v_overlap > 0.5 * min(gy2 - gy1, by2 - by1) and 0 <= bx1 - gx2 < 1.2 * h:
                    group = group + [b]
                    candidates.append(list(group))
        # vertically stacked pairs with horizontal overlap -> 2-line plates
        for i, a in enumerate(results):
            ax1, ay1, ax2, ay2 = box_of([a])
            for b in results[i + 1:]:
                bx1, by1, bx2, by2 = box_of([b])
                overlap = min(ax2, bx2) - max(ax1, bx1)
                if overlap <= 0.5 * min(ax2 - ax1, bx2 - bx1):
                    continue
                gap = max(by1, ay1) - min(by2, ay2)
                if gap < 0.8 * max(ay2 - ay1, by2 - by1):
                    candidates.append([a, b])

        best: Optional[PlateRead] = None
        for group in candidates:
            raw, conf = _merge_lines(group)
            if conf < self.min_conf:
                continue
            text, valid = normalize(raw, self.fmt)
            if len(text) < 4:
                continue
            s = score(text, conf, valid, self.fmt)
            if best is None or s > best.score:
                x1, y1, x2, y2 = box_of(group)
                bx = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
                crop = vehicle[max(bx[1], 0):bx[3], max(bx[0], 0):bx[2]]
                best = PlateRead(text, raw, conf, valid, s, box=bx, crop=crop.copy() if crop.size else None)
        return best
