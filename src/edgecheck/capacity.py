"""Does the edge survive the size you would actually run?

A backtest fills whatever it asks for. A book fills what is resting. The gap
between the two is capacity, and it is the reason a sleeve that was +222 at
$2 a side is not +1,664 at $15 a side: the extra $13 was never fillable.

Two inputs make this measurable, and most logs already have one of them:

  fill_size  what actually filled, against `size` requested -- from a live or
             paper log that recorded the book
  depth      what was resting at the touch when the order went in

Given either, each trade's P&L is rescaled to the fraction that would have
filled at a stated deploy size, and the edge is re-reported at that size.
Nothing here models market impact; it only refuses to count fills that the
recorded book could not have provided.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import Trade


@dataclass(frozen=True)
class Capacity:
    n: int                        # trades with size information
    tested_size: float            # mean requested size in the sample
    deploy_size: float            # the size the caller wants to run
    fillable_frac: float          # mean share of deploy_size the book could fill
    limited: int                  # trades where deploy_size exceeded what was available
    pnl_tested: float             # P&L as logged
    pnl_at_deploy: float          # P&L rescaled to what would have filled at deploy_size

    @property
    def limited_frac(self) -> float:
        return self.limited / self.n if self.n else 0.0

    @property
    def scale(self) -> float | None:
        """P&L at deploy size as a multiple of P&L as tested. Linear scaling
        would give deploy/tested; the shortfall from that is the capacity cost."""
        if abs(self.pnl_tested) < 1e-12:
            return None
        return self.pnl_at_deploy / self.pnl_tested

    @property
    def linear_scale(self) -> float | None:
        if self.tested_size <= 0:
            return None
        return self.deploy_size / self.tested_size

    @property
    def capped(self) -> bool:
        """Deploying larger would not have made more: over half the trades
        were depth-limited."""
        return self.limited_frac > 0.5


def _available(t: Trade) -> float | None:
    """Units the book could have filled on this trade, from whichever column
    the log carries. fill_size wins when both exist: it is observed, depth is
    a snapshot."""
    if t.fill_size is not None:
        return t.fill_size
    return t.depth


def capacity(trades: list[Trade], deploy_size: float) -> Capacity | None:
    """Rescale each trade to what `deploy_size` would have filled.

    A trade at requested size s, filled f, with P&L p, is assumed to earn p/s
    per unit. At deploy size d the fillable quantity is min(d, available), so
    the trade contributes p/s * min(d, available). The sum across trades is
    the P&L at deploy size; comparing it to linear scaling shows how much of
    the assumed edge the book could not have provided.
    """
    if deploy_size <= 0:
        return None
    rows = [(t, _available(t)) for t in trades
            if t.size is not None and t.size > 0 and _available(t) is not None]
    if not rows:
        return None
    n = len(rows)
    tested = sum(t.size or 0.0 for t, _ in rows) / n
    pnl_tested = sum(t.pnl for t, _ in rows)
    pnl_deploy = 0.0
    fillable = 0.0
    limited = 0
    for t, avail in rows:
        size = t.size or 0.0
        a = avail if avail is not None else 0.0
        fill = min(deploy_size, a)
        if a < deploy_size:
            limited += 1
        fillable += fill / deploy_size
        pnl_deploy += (t.pnl / size) * fill
    return Capacity(
        n=n, tested_size=tested, deploy_size=deploy_size,
        fillable_frac=fillable / n, limited=limited,
        pnl_tested=pnl_tested, pnl_at_deploy=pnl_deploy,
    )
