"""Per-vehicle track state and the traffic-rule engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .geometry import Line, Point
from .plate_text import PlateVoter
from .signal import RED

RED_LIGHT = "RED_LIGHT"
OVERSPEED = "OVERSPEED"
WRONG_WAY = "WRONG_WAY"


@dataclass
class Track:
    track_id: int
    cls_name: str
    first_frame: int
    last_frame: int = 0
    last_pt: Optional[Point] = None
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    t_line_a: Optional[float] = None
    t_line_b: Optional[float] = None
    speed_kmph: Optional[float] = None
    violations: List[str] = field(default_factory=list)
    voter: PlateVoter = field(default_factory=PlateVoter)
    last_ocr_frame: int = -10**9
    best_plate_crop: Optional[np.ndarray] = None
    best_plate_score: float = 0.0
    plate_box: Optional[Tuple[int, int, int, int]] = None  # last plate box, frame coordinates
    plate_box_frame: int = -10**9                          # frame it was found in

    @property
    def plate(self) -> str:
        return self.voter.best()[0]


class RuleEngine:
    def __init__(self, rules: dict, width: int, height: int):
        rl = rules.get("red_light", {}) or {}
        sp = rules.get("speed", {}) or {}
        ww = rules.get("wrong_way", {}) or {}
        self.red_enabled = bool(rl.get("enabled", False))
        self.speed_enabled = bool(sp.get("enabled", False))
        self.wrong_enabled = bool(ww.get("enabled", False))

        self.stop_line = Line.from_normalized(rl["stop_line"], width, height, "STOP LINE") \
            if self.red_enabled else None
        need_ab = self.speed_enabled or self.wrong_enabled
        self.line_a = Line.from_normalized(sp["line_a"], width, height, "A") if need_ab else None
        self.line_b = Line.from_normalized(sp["line_b"], width, height, "B") if need_ab else None
        self.distance_m = float(sp.get("distance_m", 10.0))
        self.limit = float(sp.get("limit_kmph", 50))

    def lines(self) -> List[Line]:
        return [l for l in (self.stop_line, self.line_a, self.line_b) if l is not None]

    def update(self, tr: Track, cur: Point, t: float, signal_state: str) -> List[str]:
        """Feed the new anchor point of a track; return newly triggered violations."""
        new: List[str] = []
        prev = tr.last_pt
        if prev is None:
            return new

        if self.stop_line is not None and self.stop_line.crossing(prev, cur):
            if signal_state == RED and RED_LIGHT not in tr.violations:
                new.append(RED_LIGHT)

        if self.line_a is not None:
            if tr.t_line_a is None and self.line_a.crossing(prev, cur):
                tr.t_line_a = t
            if tr.t_line_b is None and self.line_b.crossing(prev, cur):
                tr.t_line_b = t

            if tr.t_line_a is not None and tr.t_line_b is not None and tr.speed_kmph is None:
                dt = abs(tr.t_line_b - tr.t_line_a)
                if dt > 1e-3:
                    tr.speed_kmph = self.distance_m / dt * 3.6
                if self.speed_enabled and tr.speed_kmph is not None and tr.speed_kmph > self.limit:
                    new.append(OVERSPEED)
                if self.wrong_enabled and tr.t_line_b < tr.t_line_a:
                    new.append(WRONG_WAY)

        tr.violations.extend(v for v in new if v not in tr.violations)
        return new
