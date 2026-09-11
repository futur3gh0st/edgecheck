"""Does the edge survive its own costs?

This is usually the shortest path to an answer, and it is the one most often
skipped. A strategy can be genuinely right about the world -- correct direction,
honest probabilities, real edge -- and still lose money on every single trade,
because the edge it captures is smaller than the toll it pays to capture it.

When that is true, nothing fixes it. Not more capital: costs scale with size.
Not better calibration: calibration makes a model honest about a small edge, it
does not make the edge larger. Not a longer run: a negative per-unit result
compounds in the wrong direction.

So this module answers one question before any of the statistics matter:

    gross edge per unit  vs  cost per unit

If cost exceeds gross, stop. The rest of the analysis is describing exactly how
you lose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    """Where the money went, in the only three pieces that matter."""

    n: int
    gross: float          # P&L before costs
    costs: float          # everything paid to trade (positive number)
    net: float            # what actually landed

    @property
    def gross_per_trade(self) -> float:
        return self.gross / self.n if self.n else 0.0

    @property
    def cost_per_trade(self) -> float:
        return self.costs / self.n if self.n else 0.0

    @property
    def cost_ratio(self) -> float | None:
        """Costs as a fraction of gross edge.

        Above 1.0 the strategy is structurally unprofitable: it pays more to
        trade than the trade is worth. Returns None when gross is <= 0, where
        the ratio is meaningless -- there is no edge to consume.
        """
        if self.gross <= 0:
            return None
        return self.costs / self.gross

    @property
    def structurally_unprofitable(self) -> bool:
        """No amount of sizing, tuning, or patience fixes this case."""
        return self.gross <= 0 or self.costs > self.gross

    @property
    def verdict(self) -> str:
        if self.gross <= 0:
            return "NO GROSS EDGE -- loses before costs are counted"
        r = self.cost_ratio or 0.0
        if r > 1.0:
            return (f"COSTS EXCEED EDGE -- pays {r:.0%} of gross edge to trade; "
                    "not fixable by size or tuning")
        if r > 0.6:
            return f"COSTS DOMINATE -- {r:.0%} of gross edge goes to costs"
        return f"costs take {r:.0%} of gross edge"


def decompose(pnl_net: list[float], costs: list[float]) -> CostBreakdown:
    """Split realised results into gross, costs, and net.

    `pnl_net` is what landed in the account (already net of cost); `costs` is
    what was paid. Gross is recovered as net + cost rather than taken on trust,
    so the three always reconcile.
    """
    n = len(pnl_net)
    net = sum(pnl_net)
    paid = sum(abs(c) for c in costs)
    return CostBreakdown(n=n, gross=net + paid, costs=paid, net=net)


@dataclass(frozen=True)
class UnitEconomics:
    """The per-unit comparison, in whatever unit the instrument trades in.

    For binary contracts that is cents per contract. For a spot strategy it
    might be basis points per unit notional. The tool does not care which, as
    long as both sides are quoted the same way.
    """

    edge: float
    cost: float
    unit: str = "unit"

    @property
    def net(self) -> float:
        return self.edge - self.cost

    @property
    def cost_multiple(self) -> float | None:
        """How many times the edge the cost is. 2.0 means the toll is double."""
        if abs(self.edge) < 1e-12:
            return None
        return self.cost / self.edge

    def summary(self) -> str:
        m = self.cost_multiple
        mult = f"{m:.1f}x the edge" if m is not None else "edge is ~0"
        return (f"captures {self.edge:+.4f}/{self.unit}, "
                f"pays {self.cost:.4f}/{self.unit} ({mult}) "
                f"-> {self.net:+.4f}/{self.unit}")


def unit_economics(win_rate: float, breakeven: float, cost_per_unit: float,
                   unit: str = "contract") -> UnitEconomics:
    """Edge and cost per unit, for instruments priced as a probability.

    For a binary contract bought at price p, breakeven is p itself: you need to
    win that fraction of the time to break even before costs. Realised win rate
    above that price is the gross edge, in the same units as the price.
    """
    return UnitEconomics(edge=win_rate - breakeven, cost=cost_per_unit, unit=unit)
