"""Unit tests for the pure-Python parts (no models needed).

    python -m unittest discover tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr.geometry import Line  # noqa: E402
from anpr.plate_text import PlateVoter, correct_india, normalize  # noqa: E402
from anpr.signal import GREEN, RED, SignalController  # noqa: E402
from anpr.violations import OVERSPEED, RED_LIGHT, WRONG_WAY, RuleEngine, Track  # noqa: E402
from evaluate import levenshtein  # noqa: E402


class PlateTextTests(unittest.TestCase):
    def test_clean_valid(self):
        self.assertEqual(correct_india("MH 12 AB 1234"), "MH12AB1234")
        self.assertEqual(correct_india("dl3caf0001"), "DL3CAF0001")
        self.assertEqual(correct_india("KA05MN9999"), "KA05MN9999")

    def test_confusions_fixed(self):
        self.assertEqual(correct_india("MH12AB1Z34"), "MH12AB1234")   # Z -> 2 in the number
        self.assertEqual(correct_india("MHI2AB1234"), "MH12AB1234")   # I -> 1 in the district
        self.assertEqual(correct_india("KA0SMN9999"), "KA05MN9999")   # S -> 5
        self.assertEqual(correct_india("TN09B8O34"), "TN09B8034")     # O -> 0 in the number
        self.assertIsNone(correct_india("TN09B8O"))                   # too short for a plate

    def test_invalid(self):
        self.assertIsNone(correct_india("HELLO"))
        text, valid = normalize("ABC", "india")
        self.assertFalse(valid)

    def test_voter(self):
        v = PlateVoter()
        for _ in range(3):
            v.add("MH12AB1234", 0.8, True, 1.2)
        v.add("MH12AB1Z34", 0.9, False, 0.5)
        self.assertEqual(v.best()[0], "MH12AB1234")
        self.assertFalse(v.is_locked(4))
        v.add("MH12AB1234", 0.8, True, 1.2)
        self.assertTrue(v.is_locked(4))

    def test_consensus_fixes_single_char_errors(self):
        # no read is fully right, but each position is right in the majority
        v = PlateVoter()
        v.add("MH12AB1284", 0.7, True, 1.0)   # 3 -> 8
        v.add("MH12AB7234", 0.7, True, 1.0)   # 1 -> 7
        v.add("MH17AB1234", 0.7, True, 1.0)   # 2 -> 7
        v.add("MH12AB1294", 0.7, True, 1.0)   # 3 -> 9
        self.assertEqual(v.best()[0], "MH12AB1234")

    def test_consensus_needs_enough_reads(self):
        v = PlateVoter()
        v.add("MH12AB1284", 0.7, True, 1.0)
        v.add("MH12AB1234", 0.6, True, 0.9)
        self.assertEqual(v.best()[0], "MH12AB1284")   # falls back to top single read


class DeskewTests(unittest.TestCase):
    def test_straightens_tilted_plate(self):
        import cv2
        import numpy as np
        from anpr.ocr import deskew

        plate = np.full((80, 300), 255, np.uint8)
        cv2.rectangle(plate, (5, 5), (294, 74), 0, 3)
        cv2.putText(plate, "MH12AB1234", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
        canvas = np.full((160, 380), 255, np.uint8)
        canvas[40:120, 40:340] = plate
        m = cv2.getRotationMatrix2D((190, 80), -8, 1.0)
        tilted = cv2.warpAffine(canvas, m, (380, 160), borderValue=255)
        fixed = deskew(tilted)
        self.assertIsNotNone(fixed)
        self.assertIsNone(deskew(fixed))   # already level -> nothing more to do


class GeometryTests(unittest.TestCase):
    def test_crossing(self):
        ln = Line((0, 100), (200, 100))
        self.assertNotEqual(ln.crossing((50, 90), (50, 110)), 0)
        self.assertEqual(ln.crossing((50, 80), (50, 95)), 0)
        self.assertEqual(ln.crossing((250, 90), (250, 110)), 0)  # outside the segment


class RuleTests(unittest.TestCase):
    rules = {
        "red_light": {"enabled": True, "stop_line": [[0, 0.5], [1, 0.5]]},
        "speed": {"enabled": True, "line_a": [[0, 0.3], [1, 0.3]], "line_b": [[0, 0.7], [1, 0.7]],
                  "distance_m": 10, "limit_kmph": 50},
        "wrong_way": {"enabled": True},
    }

    def drive(self, ys, dt, signal):
        eng = RuleEngine(self.rules, 100, 100)
        tr = Track(1, "car", 0)
        found = []
        for i, y in enumerate(ys):
            found += eng.update(tr, (50, y), i * dt, signal)
            tr.last_pt = (50, y)
        return tr, found

    def test_red_light_and_speed(self):
        # line A (y=30) crossed at t=0.1, line B (y=70) at t=0.4 -> 10 m in 0.3 s = 120 km/h
        tr, found = self.drive([20, 40, 50 + 1, 60, 80, 90], 0.1, RED)
        self.assertIn(RED_LIGHT, found)
        self.assertIn(OVERSPEED, found)
        self.assertAlmostEqual(tr.speed_kmph, 120.0, delta=1)

    def test_green_slow(self):
        tr, found = self.drive([20, 40, 60, 80], 1.0, GREEN)
        self.assertEqual(found, [])
        self.assertLess(tr.speed_kmph, 50)

    def test_wrong_way(self):
        _, found = self.drive([90, 60, 40, 20], 1.0, GREEN)
        self.assertIn(WRONG_WAY, found)


class SignalTests(unittest.TestCase):
    def test_manual(self):
        s = SignalController({"mode": "manual", "red_intervals": [[0, 5]]})
        self.assertEqual(s.state(2), RED)
        self.assertEqual(s.state(6), GREEN)

    def test_timed(self):
        s = SignalController({"mode": "timed", "timed": {"red": 10, "green": 5, "start_red": True}})
        self.assertEqual(s.state(3), RED)
        self.assertEqual(s.state(12), GREEN)
        self.assertEqual(s.state(16), RED)


class MetricTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("MH12AB1234", "MH12AB1234"), 0)
        self.assertEqual(levenshtein("MH12AB1234", "MH12A81234"), 1)
        self.assertEqual(levenshtein("", "ABC"), 3)


if __name__ == "__main__":
    unittest.main()
