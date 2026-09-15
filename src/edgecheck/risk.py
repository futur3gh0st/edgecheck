"""Would you have survived the path to the mean?

An edge that is distinguishable from zero can still take a bankroll to zero on
the way. The mean says what the strategy pays on average; the path says what
it demands you hold through first. A strategy that is +0.5 per trade with a
drawdown of 60% of the stake is not a strategy most people can run, and no
amount of expectancy arithmetic will say so.

Everything here needs trades in entry order. Without a time column the
drawdown is over file order, which is usually entry order but not always;
the report says which it used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .stats import group_by_cluster


@dataclass(frozen=True)
class Drawdown:
    """The worst stretch of the realised path."""

    n: int
    final: float
    peak: float
    max_drawdown: float          # peak-to-trough, as a positive number
    max_drawdown_at: int         # trade index where the trough was hit
    longest_underwater: int      # most consecutive trades below a prior peak
    underwater_now: bool

    def drawdown_frac(self, bankroll: float) -> float | None:
        """Max drawdown as a share of a stated bankroll."""
        if bankroll <= 0:
            return None
        return self.max_drawdown / bankroll


def drawdown(pnl: list[float]) -> Drawdown:
    """Peak-to-trough over the cumulative path, in the order given."""
    n = len(pnl)
    if n == 0:
        return Drawdown(0, 0.0, 0.0, 0.0, 0, 0, False)
    equity = 0.0
    peak = 0.0
    worst = 0.0
    worst_at = 0
    run = 0
    longest = 0
    for i, x in enumerate(pnl):
        equity += x
        if equity >= peak:
            peak = equity
            run = 0
        else:
            run += 1
            longest = max(longest, run)
            dd = peak - equity
            if dd > worst:
                worst, worst_at = dd, i
    return Drawdown(
        n=n, final=equity, peak=peak, max_drawdown=worst, max_drawdown_at=worst_at,
        longest_underwater=longest, underwater_now=run > 0,
    )


@dataclass(frozen=True)
class Ruin:
    """Chance of exhausting a bankroll over a horizon, by resampling the path."""

    bankroll: float
    horizon: int
    reps: int
    ruined: int

    @property
    def probability(self) -> float:
        return self.ruined / self.reps if self.reps else 0.0


def ruin_probability(
    pnl: list[float],
    bankroll: float,
    clusters: list[str | None] | None = None,
    horizon: int | None = None,
    reps: int = 2000,
    seed: int = 0,
) -> Ruin | None:
    """P(cumulative loss ever reaches `bankroll`) over `horizon` trades, by
    resampling whole clusters of the observed P&L with replacement.

    This is not a model of the strategy; it is the observed distribution,
    replayed. It answers "if the future looks like the sample, how often does
    this stake go to zero?" and nothing more. Horizon defaults to the sample
    length, so the question is "how often would *this run* have ruined me".
    """
    n = len(pnl)
    if n < 2 or bankroll <= 0:
        return None
    blocks = list(group_by_cluster(pnl, clusters).values()) if clusters is not None else [[x] for x in pnl]
    if len(blocks) < 2:
        return None
    h = horizon if horizon is not None else n
    rng = random.Random(seed)
    k = len(blocks)
    ruined = 0
    for _ in range(reps):
        equity = 0.0
        drawn = 0
        dead = False
        while drawn < h and not dead:
            for x in blocks[rng.randrange(k)]:
                equity += x
                drawn += 1
                if equity <= -bankroll:
                    dead = True
                    break
                if drawn >= h:
                    break
        ruined += dead
    return Ruin(bankroll=bankroll, horizon=h, reps=reps, ruined=ruined)
