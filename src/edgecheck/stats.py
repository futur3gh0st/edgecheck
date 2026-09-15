"""Sample statistics that refuse to overclaim.

Everything here exists to answer one question honestly: is what you are looking
at a result, or is it noise that happens to be pointing in a pleasant direction?

The guards are the point. A handful of wins on high-probability bets produces a
tiny sample standard deviation, a narrow interval, and a confident-looking
verdict -- the exact shape of a strategy that is about to lose money slowly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.959963985     # two-sided 95%
Z80 = 0.8416212336    # 80% power

# Below this many resolved trades, no verdict is rendered at any confidence.
# Not a statistical constant -- a policy. The interval alone will happily call
# three lucky coin flips significant.
MIN_N_FOR_VERDICT = 30

# A bucket-level warning (overconfident, concentrated loss) is only raised when
# the bucket sits at least this many standard errors from its null. Below it
# the deviation is reported as noise, because at n=80 a 2.7-point calibration
# gap is inside one binomial SE and flagging it would be the exact overclaim
# this tool exists to prevent.
SIGNIFICANT_Z = 2.0

# Student's t, two-sided 95%, by degrees of freedom. Using the normal 1.96 at
# small n understates the interval badly: at n=3 the true multiplier is 4.30.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t95(df: int) -> float:
    """Two-sided 95% critical value; approaches the normal limit for large df."""
    if df < 1:
        return float("inf")
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k <= df]
    return _T95[max(smaller)] if smaller else Z95


def mean_sd(xs: list[float]) -> tuple[float, float]:
    """Sample mean and sample standard deviation (n-1)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


@dataclass(frozen=True)
class Expectancy:
    """What a series of outcomes actually returned, and whether we can tell."""

    n: int
    total: float
    mean: float
    sd: float
    se: float
    lo: float
    hi: float

    @property
    def underpowered(self) -> bool:
        return self.n < MIN_N_FOR_VERDICT

    @property
    def distinguishable(self) -> bool:
        """True only when the interval excludes zero AND the sample is large
        enough to be believed. Both conditions, deliberately."""
        if self.underpowered or self.se <= 0:
            return False
        return self.lo > 0 or self.hi < 0

    @property
    def verdict(self) -> str:
        if self.underpowered:
            return f"NO VERDICT -- {self.n} resolved, need {MIN_N_FOR_VERDICT}"
        if self.distinguishable:
            return "DISTINGUISHABLE from zero"
        return "NOT distinguishable from zero"


def expectancy(pnl: list[float]) -> Expectancy:
    """Per-unit expectancy with a t-interval sized for the sample actually held."""
    n = len(pnl)
    if n == 0:
        return Expectancy(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = sum(pnl)
    m, sd = mean_sd(pnl)
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    crit = t95(n - 1)
    lo, hi = (m - crit * se, m + crit * se) if se else (m, m)
    return Expectancy(n=n, total=total, mean=m, sd=sd, se=se, lo=lo, hi=hi)


def binomial_z(predicted: float, actual: float, n: int) -> float | None:
    """How many standard errors `actual` sits from `predicted`, for a win rate
    observed over n trials whose null probability is `predicted`."""
    if n <= 0 or predicted <= 0.0 or predicted >= 1.0:
        return None
    se = math.sqrt(predicted * (1.0 - predicted) / n)
    return (actual - predicted) / se


def sum_z(total: float, sd: float, n: int) -> float | None:
    """How many standard errors a bucket's summed P&L sits from zero, given the
    per-trade standard deviation of the whole sample."""
    if n <= 0 or sd <= 0.0:
        return None
    return total / (sd * math.sqrt(n))


def required_n(effect: float, sd: float, floor: int = MIN_N_FOR_VERDICT) -> int | None:
    """Resolved trades needed to detect `effect` at 95% confidence, 80% power.

    Returns None when it cannot be computed. Never returns less than `floor`:
    the arithmetic will cheerfully say "1 trade" when the standard deviation was
    itself estimated from three similar numbers.
    """
    if sd <= 0 or abs(effect) < 1e-12:
        return None
    n = ((Z95 + Z80) ** 2) * (sd ** 2) / (effect ** 2)
    return max(int(math.ceil(n)), floor)


@dataclass(frozen=True)
class Bucket:
    """One row of a calibration table or a dimensional split."""

    key: float
    predicted: float
    actual: float
    n: int
    total_pnl: float

    @property
    def gap(self) -> float:
        """Realised minus predicted. Negative means overconfident."""
        return self.actual - self.predicted

    @property
    def z(self) -> float | None:
        """Gap in binomial standard errors, taking the model's own probability
        as the null. None when the null is degenerate (p of 0 or 1) or n is 0."""
        return binomial_z(self.predicted, self.actual, self.n)

    @property
    def significant(self) -> bool:
        return self.z is not None and abs(self.z) >= SIGNIFICANT_Z


def bucket_by(
    rows: list[tuple[float, float, float, float]],
    n_buckets: int = 5,
    min_rows: int = 5,
) -> list[Bucket]:
    """Split (key, predicted, won, pnl) rows into equal-count buckets by key.

    Equal-count rather than equal-width: a strategy that only ever trades 92-98%
    probabilities would put every row in one bucket of a fixed-width grid and
    show nothing.
    """
    pts = sorted(r for r in rows if r[0] is not None)
    if not pts:
        return []

    # Ask for fewer buckets rather than emit thin ones. Splitting 22 rows five
    # ways gives chunks of four; if every chunk is then rejected as too thin the
    # table comes back empty, which reads as "no data" rather than "not enough
    # for five buckets".
    usable = max(1, min(n_buckets, len(pts) // max(1, min_rows)))
    size = max(1, len(pts) // usable)

    def make(chunk: list[tuple[float, float, float, float]]) -> Bucket:
        return Bucket(
            key=sum(c[0] for c in chunk) / len(chunk),
            predicted=sum(c[1] for c in chunk) / len(chunk),
            actual=sum(c[2] for c in chunk) / len(chunk),
            n=len(chunk),
            total_pnl=sum(c[3] for c in chunk),
        )

    out: list[Bucket] = []
    for i in range(0, len(pts), size):
        chunk = pts[i:i + size]
        if not chunk:
            continue
        # A remainder shorter than a full bucket joins the last one rather than
        # standing alone as a bucket of two.
        if out and len(chunk) < min_rows:
            prev, total = out[-1], out[-1].n + len(chunk)
            out[-1] = Bucket(
                key=(prev.key * prev.n + sum(c[0] for c in chunk)) / total,
                predicted=(prev.predicted * prev.n + sum(c[1] for c in chunk)) / total,
                actual=(prev.actual * prev.n + sum(c[2] for c in chunk)) / total,
                n=total,
                total_pnl=prev.total_pnl + sum(c[3] for c in chunk),
            )
            continue
        out.append(make(chunk))
    return out
