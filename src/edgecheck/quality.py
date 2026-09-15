"""Is the data fit to be measured?

Every statistic downstream assumes the log is what it claims to be: one row
per decision, in order, with outcomes that came after entries and a recorder
that was awake the whole time. Real logs break each of those quietly. A
recorder that slept through 87% of a window still produces a file that parses,
and a fill model run over that file produces a P&L that is an artefact of the
gaps, not a result.

So this runs first, and its findings are printed before any number that could
be mistaken for a verdict.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .loaders import ColumnMap, Trade

# A gap between consecutive entries longer than this many median spacings is
# reported. Ten is loose on purpose: a strategy that trades in bursts has long
# quiet gaps that are real, and the point is to surface recorder outages, not
# to second-guess the strategy's cadence.
GAP_MULTIPLE = 10.0

# A side split more lopsided than this is named. Not wrong in itself -- a
# one-directional strategy is a strategy -- but with `predicted_is_yes` unset
# it is the signature of P(YES) being read as P(side taken).
SIDE_IMBALANCE = 0.90

# Fraction of trades sharing a timestamp minute above which trades are called
# non-independent when no cluster column was given.
SHARED_MINUTE_FRAC = 0.10


@dataclass(frozen=True)
class Gap:
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class DataQuality:
    """What was checked and what was found. Empty lists mean clean."""

    n: int
    timed: int
    span_seconds: float | None
    median_spacing: float | None
    gaps: list[Gap] = field(default_factory=list)
    gap_seconds: float = 0.0
    duplicate_keys: int = 0
    shared_minute_frac: float = 0.0
    side_counts: dict[str, int] = field(default_factory=dict)
    clusters: int | None = None
    missing_time: int = 0
    missing_predicted: int = 0
    missing_won: int = 0

    @property
    def gap_frac(self) -> float:
        """Share of the recorded span that fell inside a reported gap."""
        if not self.span_seconds:
            return 0.0
        return self.gap_seconds / self.span_seconds

    @property
    def lopsided(self) -> bool:
        total = sum(self.side_counts.values())
        if total < 20 or len(self.side_counts) < 2:
            return False
        return max(self.side_counts.values()) / total >= SIDE_IMBALANCE

    @property
    def dependent_without_cluster(self) -> bool:
        """Trades share timestamps but no cluster was named. The naive CI is
        then an overclaim, and the verdict downstream is capped."""
        return self.clusters is None and self.shared_minute_frac >= SHARED_MINUTE_FRAC

    @property
    def findings(self) -> list[str]:
        out: list[str] = []
        if self.gaps:
            out.append(
                f"{len(self.gaps)} gap(s) > {GAP_MULTIPLE:.0f}x median spacing, "
                f"covering {self.gap_frac:.0%} of the recorded span "
                f"(longest {_human(max(g.seconds for g in self.gaps))}) -- "
                "was the recorder running?"
            )
        if self.duplicate_keys:
            out.append(f"{self.duplicate_keys} duplicate key(s): the same trade counted twice?")
        if self.dependent_without_cluster:
            out.append(
                f"{self.shared_minute_frac:.0%} of trades share an entry minute and no "
                "--cluster was given: they are not independent evidence"
            )
        if self.lopsided:
            big = max(self.side_counts, key=lambda k: self.side_counts[k])
            out.append(
                f"{self.side_counts[big] / sum(self.side_counts.values()):.0%} of trades are "
                f"'{big}' -- if the model logs P(YES), pass --predicted-is-yes"
            )
        if self.missing_time:
            out.append(f"{self.missing_time} row(s) have an unreadable time and were "
                       "placed last")
        if self.missing_predicted:
            out.append(f"{self.missing_predicted} row(s) lack a prediction and are "
                       "excluded from calibration")
        if self.missing_won:
            out.append(f"{self.missing_won} row(s) lack an outcome and are excluded "
                       "from calibration")
        return out

    @property
    def clean(self) -> bool:
        return not self.findings


def _human(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def assess(trades: list[Trade], m: ColumnMap) -> DataQuality:
    """Run every check the mapping makes possible. Trades are expected in the
    order `load` returns them (entry order when a time column exists)."""
    n = len(trades)
    times = [t.time for t in trades if t.time is not None]
    timed = len(times)

    span: float | None = None
    spacing: float | None = None
    gaps: list[Gap] = []
    shared_frac = 0.0
    if timed >= 2:
        ts = sorted(times)
        span = ts[-1] - ts[0]
        diffs = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        positive = [d for d in diffs if d > 0]
        spacing = _median(positive) if positive else 0.0
        if spacing > 0:
            limit = GAP_MULTIPLE * spacing
            gaps = [Gap(a, b) for a, b in zip(ts, ts[1:], strict=False) if b - a > limit]
        minutes = Counter(math.floor(t / 60.0) for t in ts)
        shared = sum(c for c in minutes.values() if c > 1)
        shared_frac = shared / timed

    dupes = 0
    if m.key:
        keys = Counter(str(t.raw.get(m.key)) for t in trades if t.raw.get(m.key) is not None)
        dupes = sum(c - 1 for c in keys.values() if c > 1)

    sides: dict[str, int] = {}
    if m.side:
        sides = dict(Counter(str(t.raw.get(m.side, "")).strip().lower()
                             for t in trades if t.raw.get(m.side) not in (None, "")))

    clusters: int | None = None
    if m.cluster:
        clusters = len({t.cluster for t in trades if t.cluster is not None})

    return DataQuality(
        n=n,
        timed=timed,
        span_seconds=span,
        median_spacing=spacing,
        gaps=gaps,
        gap_seconds=sum(g.seconds for g in gaps),
        duplicate_keys=dupes,
        shared_minute_frac=shared_frac,
        side_counts=sides,
        clusters=clusters,
        missing_time=(n - timed) if m.time else 0,
        missing_predicted=sum(1 for t in trades if t.predicted is None) if m.predicted else 0,
        missing_won=sum(1 for t in trades if t.won is None) if m.won else 0,
    )
