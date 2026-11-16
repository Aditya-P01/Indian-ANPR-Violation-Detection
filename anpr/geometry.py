"""Small geometry helpers: virtual lines and line-crossing tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float]


def _orient(a: Point, b: Point, c: Point) -> float:
    """Signed area of triangle abc (>0 counter-clockwise, <0 clockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 intersects segment q1-q2 (touching counts)."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True

    def on_seg(a: Point, b: Point, c: Point) -> bool:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    if d1 == 0 and on_seg(q1, q2, p1):
        return True
    if d2 == 0 and on_seg(q1, q2, p2):
        return True
    if d3 == 0 and on_seg(p1, p2, q1):
        return True
    if d4 == 0 and on_seg(p1, p2, q2):
        return True
    return False


@dataclass
class Line:
    """A virtual line in pixel coordinates."""

    p1: Point
    p2: Point
    name: str = ""

    @classmethod
    def from_normalized(cls, pts: Sequence[Sequence[float]], width: int, height: int, name: str = "") -> "Line":
        (x1, y1), (x2, y2) = pts
        return cls((x1 * width, y1 * height), (x2 * width, y2 * height), name)

    def side(self, pt: Point) -> int:
        v = _orient(self.p1, self.p2, pt)
        return 1 if v > 0 else (-1 if v < 0 else 0)

    def crossing(self, prev: Point, cur: Point) -> int:
        """0 if the movement prev->cur does not cross the line.

        Otherwise returns the side the object ended on (+1 / -1), which gives
        the crossing direction.
        """
        if prev is None or cur is None:
            return 0
        s_prev, s_cur = self.side(prev), self.side(cur)
        if s_prev == s_cur or s_cur == 0:
            return 0
        if not segments_intersect(prev, cur, self.p1, self.p2):
            return 0
        return s_cur

    def as_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (int(self.p1[0]), int(self.p1[1])), (int(self.p2[0]), int(self.p2[1]))
