"""Traffic-signal state: manual schedule, fixed cycle, or colour detection."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

RED, GREEN, UNKNOWN = "RED", "GREEN", "UNKNOWN"


class SignalController:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "manual")
        self.red_intervals = cfg.get("red_intervals", []) or []
        timed = cfg.get("timed", {}) or {}
        self.t_red = float(timed.get("red", 15))
        self.t_green = float(timed.get("green", 15))
        self.start_red = bool(timed.get("start_red", True))
        self.roi = cfg.get("light_roi")
        self._last_auto = UNKNOWN

    def state(self, t: float, frame: Optional[np.ndarray] = None) -> str:
        if self.mode == "manual":
            return RED if any(a <= t < b for a, b in self.red_intervals) else GREEN
        if self.mode == "timed":
            cycle = self.t_red + self.t_green
            phase = t % cycle
            first, second = (RED, GREEN) if self.start_red else (GREEN, RED)
            first_len = self.t_red if self.start_red else self.t_green
            return first if phase < first_len else second
        if self.mode == "auto" and frame is not None and self.roi:
            return self._detect(frame)
        return UNKNOWN

    def _detect(self, frame: np.ndarray) -> str:
        """Classify the lamp colour inside the ROI by counting bright red/green pixels."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        roi = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        if roi.size == 0:
            return self._last_auto
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 90, 140), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 140), (180, 255, 255))
        green = cv2.inRange(hsv, (45, 90, 140), (95, 255, 255))
        n_red, n_green = int(np.count_nonzero(red)), int(np.count_nonzero(green))
        min_px = max(8, int(0.003 * roi.shape[0] * roi.shape[1]))
        if max(n_red, n_green) >= min_px:
            self._last_auto = RED if n_red > n_green else GREEN
        return self._last_auto
