"""Clean-up, format correction, scoring and multi-frame voting for plate text.

OCR engines often confuse look-alike characters (O/0, I/1, B/8, S/5 ...).
Number plates follow a fixed pattern, so we can fix a read using the
position of each character: e.g. in an Indian plate the first two
characters are always letters and the last four are always digits.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7",
            "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "Y": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}

# Indian state / UT registration codes.
INDIA_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH",
}

# SS DD [L..LLL] NNNN   e.g. MH12AB1234, DL3CAF0001, KA05MN9999, TN09B1234
INDIA_FULL = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")


def clean(text: str) -> str:
    """Uppercase and keep only A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _as_letters(s: str) -> Tuple[str, int]:
    out, changes = [], 0
    for ch in s:
        if ch.isdigit():
            ch = TO_LETTER.get(ch, ch)
            changes += 1
        out.append(ch)
    return "".join(out), changes


def _as_digits(s: str) -> Tuple[str, int]:
    out, changes = [], 0
    for ch in s:
        if ch.isalpha():
            ch = TO_DIGIT.get(ch, ch)
            changes += 1
        out.append(ch)
    return "".join(out), changes


def correct_india(text: str) -> Optional[str]:
    """Return the most plausible Indian-format plate for `text`, or None.

    Tries every split  state(2) + district(1-2) + series(0-3) + number(4)
    and keeps the valid candidate that needed the fewest character swaps.
    """
    s = clean(text)
    if not 6 <= len(s) <= 11:
        return None
    best: Optional[Tuple[int, int, str]] = None
    for d_len in (2, 1):
        series_len = len(s) - 2 - d_len - 4
        if not 0 <= series_len <= 3:
            continue
        state, c1 = _as_letters(s[0:2])
        district, c2 = _as_digits(s[2:2 + d_len])
        series, c3 = _as_letters(s[2 + d_len:2 + d_len + series_len])
        number, c4 = _as_digits(s[-4:])
        cand = state + district + series + number
        if not INDIA_FULL.match(cand):
            continue
        # a real plate always starts with a real state / UT code
        if state not in INDIA_STATE_CODES:
            continue
        changes = c1 + c2 + c3 + c4
        # every state uses 2-digit district codes except Delhi, which also has 1-digit ones (DL3C...)
        if d_len == 1 and state != "DL":
            continue
        key = (changes, 0 if d_len == 2 else 1, cand)
        if best is None or key < best:
            best = key
    return best[2] if best else None


def normalize(text: str, fmt: str = "india") -> Tuple[str, bool]:
    """Return (plate_text, is_valid_format)."""
    s = clean(text)
    if fmt == "india":
        fixed = correct_india(s)
        if fixed:
            return fixed, True
        return s, False
    return s, 4 <= len(s) <= 10


def score(text: str, conf: float, valid: bool, fmt: str = "india") -> float:
    """Single-read quality score used for voting."""
    if not text:
        return 0.0
    s = conf
    if valid:
        s *= 1.5
        if fmt == "india" and text[:2] in INDIA_STATE_CODES:
            s *= 1.2
    elif len(text) < 4:
        s *= 0.2
    else:
        s *= 0.6
    return s


@dataclass
class PlateVoter:
    """Accumulates OCR reads for one tracked vehicle and picks a winner."""

    totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    best_conf: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    valid: Dict[str, bool] = field(default_factory=dict)
    fmt: str = "india"
    min_consensus_reads: int = 3

    def add(self, text: str, conf: float, valid: bool, weight: float) -> None:
        if not text or weight <= 0:
            return
        self.totals[text] += weight
        self.counts[text] += 1
        self.best_conf[text] = max(self.best_conf[text], conf)
        self.valid[text] = valid

    @property
    def n_reads(self) -> int:
        return sum(self.counts.values())

    def _top_single(self) -> str:
        # a read matching the plate format always beats free text (stickers, bus livery, shop signs)
        valid = [k for k in self.totals if self.valid.get(k)]
        pool = valid or list(self.totals)
        return max(pool, key=lambda k: self.totals[k])

    def consensus(self) -> Optional[str]:
        """Character-by-character weighted vote over valid reads of the same length.

        e.g. reads MH12AB1Z34 (x2, invalid ignored), MH12A81234, MH12AB1284, MH12AB1234
        -> every position is decided separately -> MH12AB1234.
        Returns None when there are too few reads or the result is not a valid plate.
        """
        valid = [t for t in self.totals if self.valid.get(t)]
        if not valid:
            return None
        by_len: Dict[int, float] = defaultdict(float)
        for t in valid:
            by_len[len(t)] += self.totals[t]
        length = max(by_len, key=lambda k: by_len[k])
        same = [t for t in valid if len(t) == length]
        if sum(self.counts[t] for t in same) < self.min_consensus_reads:
            return None
        chars = []
        for i in range(length):
            votes: Dict[str, float] = defaultdict(float)
            for t in same:
                votes[t[i]] += self.totals[t]
            chars.append(max(votes, key=lambda k: votes[k]))
        cand = "".join(chars)
        if self.fmt == "india" and (not INDIA_FULL.match(cand) or cand[:2] not in INDIA_STATE_CODES):
            return None
        return cand

    def best(self) -> Tuple[str, float, int]:
        """(text, best_confidence, number_of_supporting_reads)."""
        if not self.totals:
            return "", 0.0, 0
        cons = self.consensus()
        if cons:
            same = [t for t in self.totals if self.valid.get(t) and len(t) == len(cons)]
            # support = reads that agree with the consensus in at least all but one position
            support = sum(self.counts[t] for t in same
                          if sum(a != b for a, b in zip(t, cons)) <= 1)
            conf = self.best_conf.get(cons) or max(self.best_conf[t] for t in same)
            return cons, conf, support
        text = self._top_single()
        return text, self.best_conf[text], self.counts[text]

    def is_locked(self, min_agree: int = 4) -> bool:
        """True once a valid plate has been read consistently enough to stop OCR."""
        if not self.totals:
            return False
        text = self._top_single()
        return self.valid.get(text, False) and self.counts[text] >= min_agree

    def top(self, k: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.totals.items(), key=lambda kv: -kv[1])[:k]
