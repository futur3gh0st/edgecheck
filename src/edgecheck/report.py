"""Render the analysis as text a person will actually read.

Ordered by what kills a strategy fastest. Costs first, because a strategy whose
toll exceeds its edge is finished regardless of what the statistics say, and
finding that out takes one line of arithmetic rather than a backtest.
"""

from __future__ import annotations

from .fees import CostBreakdown, UnitEconomics
from .loaders import Trade
from .stats import Bucket, Expectancy, bucket_by, expectancy, required_n

BAR = "  " + "-" * 62


def _fmt(x: float | None, spec: str = "+,.2f") -> str:
    return "n/a" if x is None else format(x, spec)


def costs_section(cb: CostBreakdown, ue: UnitEconomics | None) -> list[str]:
    out = ["", "  1. COSTS -- does the edge survive the toll?", BAR,
           f"  gross, before costs   {_fmt(cb.gross)}",
           f"  costs paid            {_fmt(-cb.costs)}",
           f"  net                   {_fmt(cb.net)}"]
    r = cb.cost_ratio
    if r is not None:
        out.append(f"  costs / gross edge    {r:.0%}")
    if ue is not None:
        out.append(f"  per unit              {ue.summary()}")
    out.append(f"  verdict               {cb.verdict}")
    if cb.structurally_unprofitable:
        out += ["",
                "  >>> More capital scales costs too. Better calibration makes the",
                "      model honest about a small edge, it does not enlarge it.",
                "      Stop here unless the cost side can change."]
    return out


def expectancy_section(e: Expectancy, claimed_total: float | None) -> list[str]:
    out = ["", "  2. EXPECTANCY -- is this a result, or is it noise?", BAR,
           f"  resolved trades       {e.n:,}",
           f"  total                 {_fmt(e.total)}",
           f"  mean per trade        {_fmt(e.mean, '+.4f')}",
           f"  std dev               {_fmt(e.sd, ',.4f')}"]
    if e.se:
        out.append(f"  95% CI (t, df={e.n - 1})     [{e.lo:+.4f}, {e.hi:+.4f}]")
    out.append(f"  verdict               {e.verdict}")
    if e.underpowered:
        out.append("                        (a few wins on high-probability bets "
                   "look identical to skill)")
    if claimed_total is not None:
        out += ["", f"  model claimed         {_fmt(claimed_total)}",
                f"  actually delivered    {_fmt(e.total)}"]
        if abs(claimed_total) > 1e-9:
            out.append(f"  edge realisation      {e.total / claimed_total:.1%}"
                       "   (100% = the model was right)")
    return out


def calibration_section(buckets: list[Bucket], overall: tuple[float, float, int] | None) -> list[str]:
    out = ["", "  3. CALIBRATION -- where is the model wrong?", BAR]
    if not buckets:
        out.append("  no predicted/won pairs -- cannot calibrate.")
        return out
    out.append(f"  {'model says':>12} {'actually won':>14} {'n':>7} {'gap':>9}")
    for b in buckets:
        out.append(f"  {b.predicted:11.1%} {b.actual:13.1%} {b.n:7,} {b.gap:+8.1%}")
    if overall:
        pred, act, n = overall
        out.append(f"  {'OVERALL':>12}")
        out.append(f"  {pred:11.1%} {act:13.1%} {n:7,} {act - pred:+8.1%}")
    worst = min(buckets, key=lambda b: b.gap)
    if worst.gap < -0.02:
        out.append(f"  >>> Overconfident: worst bucket says {worst.predicted:.1%}, "
                   f"wins {worst.actual:.1%}.")
    return out


def split_section(buckets: list[Bucket], column: str) -> list[str]:
    out = ["", f"  4. SPLIT BY {column.upper()} -- is the loss concentrated?", BAR,
           f"  {column:>14} {'model':>8} {'actual':>8} {'gap':>8} {'n':>7} {'pnl':>11}"]
    for b in buckets:
        pred = f"{b.predicted:7.1%}" if b.predicted else "      -"
        act = f"{b.actual:7.1%}" if b.actual else "      -"
        gap = f"{b.gap:+7.1%}" if b.predicted else "      -"
        out.append(f"  {b.key:14,.2f} {pred} {act} {gap} {b.n:7,} {b.total_pnl:+11,.2f}")
    if len(buckets) > 1:
        worst = min(buckets, key=lambda b: b.total_pnl)
        rest = sum(b.total_pnl for b in buckets if b is not worst)
        if worst.total_pnl < 0 < rest:
            out += ["",
                    f"  >>> One bucket ({column}~{worst.key:,.2f}) loses {worst.total_pnl:+,.2f} "
                    f"while the rest make {rest:+,.2f}.",
                    "      Cutting it AFTER seeing this is in-sample fitting. Test it",
                    "      out-of-sample before believing it."]
    return out


def power_section(e: Expectancy, claimed_per_trade: float | None) -> list[str]:
    out = ["", "  5. POWER -- how much data would settle it?", BAR]
    target = abs(claimed_per_trade) if claimed_per_trade else abs(e.mean)
    need = required_n(target, e.sd)
    if need is None:
        out.append("  not computable yet -- need more resolved trades.")
        return out
    out.append(f"  to detect {target:.4f}/trade at 95% conf, 80% power:")
    out.append(f"  trades needed         {need:,}")
    out.append(f"  have                  {e.n:,}")
    if e.n < need:
        out.append(f"  >>> UNDERPOWERED: {need - e.n:,} more needed.")
    else:
        out.append("  adequately powered.")
    return out


def render(trades: list[Trade], split_by: str | None = None,
           unit_econ: UnitEconomics | None = None, buckets: int = 5) -> str:
    """The whole report, in the order that reaches a decision fastest."""
    from .fees import decompose

    if not trades:
        return "\n  No resolved trades found. Nothing to measure.\n"

    pnl = [t.pnl for t in trades]
    e = expectancy(pnl)
    cb = decompose(pnl, [t.cost for t in trades])

    claimed = [t.claimed_value for t in trades if t.claimed_value is not None]
    claimed_total = sum(claimed) if claimed else None
    claimed_each = (claimed_total / len(claimed)) if claimed else None

    lines = [f"  trades {len(trades):,}"]
    lines += costs_section(cb, unit_econ)
    lines += expectancy_section(e, claimed_total)

    cal = [(t.predicted, t.predicted, 1.0 if t.won else 0.0, t.pnl)
           for t in trades if t.predicted is not None and t.won is not None]
    if cal:
        bs = bucket_by(cal, buckets)
        overall = (sum(c[1] for c in cal) / len(cal),
                   sum(c[2] for c in cal) / len(cal), len(cal))
        lines += calibration_section(bs, overall)
    else:
        lines += calibration_section([], None)

    if split_by:
        rows = []
        for t in trades:
            v = t.raw.get(split_by)
            try:
                key = float(v)
            except (TypeError, ValueError):
                continue
            rows.append((key,
                         t.predicted if t.predicted is not None else 0.0,
                         1.0 if t.won else 0.0,
                         t.pnl))
        if rows:
            lines += split_section(bucket_by(rows, buckets), split_by)
        else:
            lines += ["", f"  4. SPLIT BY {split_by.upper()}", BAR,
                      f"  column '{split_by}' is absent or non-numeric."]

    lines += power_section(e, claimed_each)
    return "\n".join(lines) + "\n"
