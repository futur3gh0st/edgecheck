"""Compute the analysis once; render it as text or as data.

Ordered by what kills a strategy fastest. Data quality first, because a
statistic on a log with holes in it is a statistic about the holes. Costs
next, because a strategy whose toll exceeds its edge is finished regardless
of what the statistics say. Then whether the result is distinguishable from
noise, where the model is wrong, whether the loss is concentrated, whether
the path was survivable, whether the edge survives deploy size, how much
more data would settle it -- and one verdict, with the rule that produced it.

`analyse()` does the arithmetic and returns an `Analysis`; `format_report()`
renders it and `to_dict()` serialises it. Nothing is computed while printing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capacity import Capacity, capacity
from .fees import CostBreakdown, UnitEconomics, decompose
from .loaders import ColumnMap, Trade, iter_numeric
from .quality import DataQuality, assess
from .risk import Drawdown, Ruin, drawdown, ruin_probability
from .stats import (
    SIGNIFICANT_Z,
    Bucket,
    Expectancy,
    block_bootstrap,
    bucket_by,
    expectancy,
    holm,
    required_n,
    sum_z,
    z_to_p,
)
from .verdict import Rule, Verdict, decide

BAR = "  " + "-" * 62
VERSION = "0.2.0"


@dataclass(frozen=True)
class Options:
    """Everything the CLI can ask for beyond the column mapping."""

    split_by: str | None = None
    buckets: int = 5
    target: float | None = None
    unit_econ: UnitEconomics | None = None
    bankroll: float | None = None
    deploy_size: float | None = None
    bootstrap: bool = False
    rule: Rule | None = None
    min_units: int | None = None


@dataclass(frozen=True)
class SplitResult:
    column: str
    buckets: list[Bucket]
    z: list[float | None]
    p_adjusted: list[float | None]


@dataclass(frozen=True)
class Analysis:
    n: int
    quality: DataQuality
    costs: CostBreakdown
    unit_econ: UnitEconomics | None
    expectancy: Expectancy
    naive: Expectancy | None                 # per-trade interval when clustered, for comparison
    bootstrap: tuple[float, float] | None
    claimed_total: float | None
    claimed_each: float | None
    calibration: list[Bucket]
    calibration_p: list[float | None]
    calibration_overall: tuple[float, float, int] | None
    split: SplitResult | None
    drawdown: Drawdown | None
    ruin: Ruin | None
    bankroll: float | None
    capacity: Capacity | None
    power_target: float | None
    power_needed: int | None
    verdict: Verdict
    ordered_by_time: bool = False
    notes: list[str] = field(default_factory=list)


def analyse(trades: list[Trade], m: ColumnMap | None = None, opt: Options | None = None) -> Analysis | None:
    """The whole computation. Returns None when there is nothing to measure."""
    if not trades:
        return None
    m = m or ColumnMap()
    opt = opt or Options()

    pnl = [t.pnl for t in trades]
    clustered = bool(m.cluster)
    clusters = [t.cluster for t in trades] if clustered else None

    quality = assess(trades, m)
    costs = decompose(pnl, [t.cost for t in trades])
    e = expectancy(pnl, clusters, min_n=opt.min_units)
    naive = expectancy(pnl) if clustered else None
    boot = block_bootstrap(pnl, clusters) if opt.bootstrap else None

    claimed = [v for t in trades if (v := t.claimed_value) is not None]
    claimed_total = sum(claimed) if claimed else None
    claimed_each = claimed_total / len(claimed) if claimed_total is not None else None

    cal_rows = [(t.predicted, t.predicted, 1.0 if t.won else 0.0, t.pnl)
                for t in trades if t.predicted is not None and t.won is not None]
    cal: list[Bucket] = []
    cal_p: list[float | None] = []
    cal_overall: tuple[float, float, int] | None = None
    if cal_rows:
        cal = bucket_by(cal_rows, opt.buckets)
        cal_p = _adjusted([b.z for b in cal])
        cal_overall = (sum(c[1] for c in cal_rows) / len(cal_rows),
                       sum(c[2] for c in cal_rows) / len(cal_rows), len(cal_rows))

    split: SplitResult | None = None
    if opt.split_by:
        rows = [(key, t.predicted if t.predicted is not None else 0.0,
                 1.0 if t.won else 0.0, t.pnl)
                for key, t in iter_numeric(trades, opt.split_by)]
        if rows:
            bs = bucket_by(rows, opt.buckets)
            zs = [sum_z(b.total_pnl, e.sd, b.n) for b in bs]
            split = SplitResult(opt.split_by, bs, zs, _adjusted(zs))
        else:
            split = SplitResult(opt.split_by, [], [], [])

    timed = bool(m.time) and all(t.time is not None for t in trades)
    dd = drawdown(pnl)
    ruin = (ruin_probability(pnl, opt.bankroll, clusters)
            if opt.bankroll is not None and opt.bankroll > 0 else None)

    cap = capacity(trades, opt.deploy_size) if opt.deploy_size is not None else None

    target = claimed_each if claimed_each is not None else opt.target
    needed = required_n(abs(target), e.sd) if target is not None else None

    verdict = decide(e, costs, quality=quality, dd=dd, bankroll=opt.bankroll, rule=opt.rule)

    notes: list[str] = []
    if not timed:
        notes.append("no --time column: drawdown is over file order")

    return Analysis(
        n=len(trades), quality=quality, costs=costs, unit_econ=opt.unit_econ,
        expectancy=e, naive=naive, bootstrap=boot,
        claimed_total=claimed_total, claimed_each=claimed_each,
        calibration=cal, calibration_p=cal_p, calibration_overall=cal_overall,
        split=split, drawdown=dd, ruin=ruin, bankroll=opt.bankroll, capacity=cap,
        power_target=target, power_needed=needed, verdict=verdict,
        ordered_by_time=timed, notes=notes,
    )


def _adjusted(zs: list[float | None]) -> list[float | None]:
    """Holm-adjusted p-values aligned with `zs`; None where z was None."""
    present = [(i, z) for i, z in enumerate(zs) if z is not None]
    idx = [i for i, _ in present]
    adj = holm([z_to_p(z) for _, z in present])
    out: list[float | None] = [None] * len(zs)
    for i, p in zip(idx, adj, strict=True):
        out[i] = p
    return out


# ---- text ------------------------------------------------------------------

def _fmt(x: float | None, spec: str = "+,.2f") -> str:
    return "n/a" if x is None else format(x, spec)


def _z(z: float | None, digits: int = 1) -> str:
    """Column form of a z-score; verdict lines use two decimals so a bucket at
    z=-1.95 is not printed as -2.0 beside the words '|z| < 2'."""
    return "    -" if z is None else f"{z:+5.{digits}f}"


def _p(p: float | None) -> str:
    return "     -" if p is None else f"{p:6.3f}"


def quality_section(q: DataQuality) -> list[str]:
    out = ["", "  0. DATA -- is this fit to be measured?", BAR]
    if q.timed:
        span = q.span_seconds or 0.0
        out.append(f"  span                  {span / 86400:.1f} days, {q.timed:,} timed rows")
        if q.median_spacing:
            out.append(f"  median spacing        {q.median_spacing:,.0f}s")
    if q.clusters is not None:
        out.append(f"  clusters              {q.clusters:,}")
    if q.clean:
        out.append("  findings              none")
    else:
        out.append("  findings")
        for f in q.findings:
            out.append(f"    ! {f}")
    return out


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


def expectancy_section(e: Expectancy, naive: Expectancy | None,
                       boot: tuple[float, float] | None,
                       claimed_total: float | None, dependent: bool) -> list[str]:
    out = ["", "  2. EXPECTANCY -- is this a result, or is it noise?", BAR,
           f"  resolved trades       {e.n:,}"]
    if e.clusters is not None:
        out.append(f"  clusters              {e.clusters:,}   (the unit of evidence)")
    out += [f"  total                 {_fmt(e.total)}",
            f"  mean per trade        {_fmt(e.mean, '+.4f')}",
            f"  std dev               {_fmt(e.sd, ',.4f')}"]
    if e.se:
        df = (e.clusters - 1) if e.clusters is not None else (e.n - 1)
        kind = "cluster-robust" if e.clusters is not None else "t"
        label = f"95% CI ({kind}, df={df})"
        out.append(f"  {label:<22}[{e.lo:+.4f}, {e.hi:+.4f}]")
    if naive is not None and naive.se:
        out.append(f"  naive per-trade CI    [{naive.lo:+.4f}, {naive.hi:+.4f}]"
                   "   (what 0.1 would have printed)")
    if dependent and e.se:
        out.append("                        ^ NAIVE: trades share timestamps and no "
                   "--cluster was given")
    if boot is not None:
        out.append(f"  95% bootstrap         [{boot[0]:+.4f}, {boot[1]:+.4f}]   (blocked by cluster)")
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


def calibration_section(buckets: list[Bucket], p_adj: list[float | None],
                        overall: tuple[float, float, int] | None) -> list[str]:
    """Bucket table plus a verdict on the worst bucket, Holm-adjusted for the
    number of buckets looked at."""
    out = ["", "  3. CALIBRATION -- where is the model wrong?", BAR]
    if not buckets:
        out.append("  no predicted/won pairs -- cannot calibrate.")
        return out
    out.append(f"  {'model says':>12} {'actually won':>14} {'n':>7} {'gap':>9} {'z':>6} {'p adj':>7}")
    for b, p in zip(buckets, p_adj, strict=True):
        out.append(f"  {b.predicted:11.1%} {b.actual:13.1%} {b.n:7,} {b.gap:+8.1%} "
                   f"{_z(b.z):>6} {_p(p):>7}")
    if overall:
        pred, act, n = overall
        out.append(f"  {'OVERALL':>12}")
        out.append(f"  {pred:11.1%} {act:13.1%} {n:7,} {act - pred:+8.1%}")
    worst_i = min(range(len(buckets)), key=lambda i: buckets[i].gap)
    worst, p = buckets[worst_i], p_adj[worst_i]
    where = (f"bucket says {worst.predicted:.1%}, wins {worst.actual:.1%} "
             f"(n={worst.n}, z={_z(worst.z, 2).strip()}, p={_p(p).strip()} after Holm)")
    if worst.gap < 0 and p is not None and p < 0.05:
        out.append(f"  >>> Overconfident: {where}.")
    elif worst.gap < 0:
        out.append(f"  worst {where}: within noise across {len(buckets)} buckets.")
    return out


def split_section(s: SplitResult, sd: float) -> list[str]:
    out = ["", f"  4. SPLIT BY {s.column.upper()} -- is the loss concentrated?", BAR]
    if not s.buckets:
        out.append(f"  column '{s.column}' is absent or non-numeric.")
        return out
    out.append(f"  {s.column:>14} {'model':>8} {'actual':>8} {'gap':>8} {'n':>7} "
               f"{'pnl':>11} {'z':>6} {'p adj':>7}")
    for b, z, p in zip(s.buckets, s.z, s.p_adjusted, strict=True):
        has_model = b.predicted > 0
        pred = f"{b.predicted:7.1%}" if has_model else "      -"
        act = f"{b.actual:7.1%}" if has_model else "      -"
        gap = f"{b.gap:+7.1%}" if has_model else "      -"
        out.append(f"  {b.key:14,.2f} {pred} {act} {gap} {b.n:7,} {b.total_pnl:+11,.2f} "
                   f"{_z(z):>6} {_p(p):>7}")
    if len(s.buckets) > 1:
        worst_i = min(range(len(s.buckets)), key=lambda i: s.buckets[i].total_pnl)
        worst, z, p = s.buckets[worst_i], s.z[worst_i], s.p_adjusted[worst_i]
        rest = sum(b.total_pnl for i, b in enumerate(s.buckets) if i != worst_i)
        if worst.total_pnl < 0 < rest:
            where = (f"bucket ({s.column}~{worst.key:,.2f}) loses {worst.total_pnl:+,.2f} "
                     f"while the rest make {rest:+,.2f} (z={_z(z, 2).strip()}, "
                     f"p={_p(p).strip()} after Holm)")
            if p is not None and p < 0.05:
                out += ["", f"  >>> One {where}.",
                        "      Cutting it AFTER seeing this is in-sample fitting. Test it",
                        "      out-of-sample before believing it."]
            else:
                out += ["", f"  worst {where}: within noise across {len(s.buckets)} buckets."]
    return out


def risk_section(dd: Drawdown, ruin: Ruin | None, bankroll: float | None,
                 ordered: bool) -> list[str]:
    out = ["", "  5. PATH -- would you have survived the way there?", BAR,
           f"  final                 {_fmt(dd.final)}",
           f"  peak                  {_fmt(dd.peak)}",
           f"  max drawdown          {_fmt(-dd.max_drawdown)}   at trade {dd.max_drawdown_at + 1:,}",
           f"  longest underwater    {dd.longest_underwater:,} trades"
           + ("   (still under water at the end)" if dd.underwater_now else "")]
    if bankroll is not None and bankroll > 0:
        frac = dd.drawdown_frac(bankroll) or 0.0
        out.append(f"  vs bankroll {_fmt(bankroll, ',.0f'):>9}  drawdown took {frac:.0%}")
        if ruin is not None:
            out.append(f"  P(ruin)               {ruin.probability:.1%} over {ruin.horizon:,} trades, "
                       f"{ruin.reps:,} resampled paths")
    else:
        out.append("  pass --bankroll to see the drawdown as a share of a stake and P(ruin)")
    if not ordered:
        out.append("  (no --time column: path is file order, which may not be entry order)")
    return out


def capacity_section(c: Capacity | None, deploy: float | None) -> list[str]:
    out = ["", "  6. CAPACITY -- does the edge survive the size you would run?", BAR]
    if deploy is None:
        out.append("  pass --deploy-size with --size and --fill-size or --depth to check")
        return out
    if c is None:
        out.append("  no rows carry both a size and a fill_size/depth -- cannot check.")
        return out
    out += [f"  tested at             {c.tested_size:,.2f} per trade, {c.n:,} trades",
            f"  deploy at             {c.deploy_size:,.2f}",
            f"  fillable              {c.fillable_frac:.0%} of deploy size on average",
            f"  depth-limited         {c.limited:,} of {c.n:,} trades ({c.limited_frac:.0%})",
            f"  P&L as tested         {_fmt(c.pnl_tested)}",
            f"  P&L at deploy size    {_fmt(c.pnl_at_deploy)}"]
    if c.linear_scale is not None:
        out.append(f"  linear scaling says   {_fmt(c.pnl_tested * c.linear_scale)}"
                   f"   ({c.linear_scale:.1f}x)")
    if c.capped:
        out += ["", "  >>> The book does not deepen because you would like it to.",
                "      Over half the trades were depth-limited: deploying larger",
                "      would not have made more."]
    return out


def power_section(e: Expectancy, target: float | None, need: int | None) -> list[str]:
    out = ["", "  7. POWER -- how much data would settle it?", BAR]
    if target is None:
        out += ["  no target edge. Pass --claimed (edge the model asserted at entry)",
                "  or --target EDGE (per trade) to size the run. The observed mean is",
                "  not used as the target: that only restates the p-value above."]
        return out
    if need is None:
        out.append("  not computable yet -- need more resolved trades.")
        return out
    out.append(f"  to detect {abs(target):.4f}/trade at 95% conf, 80% power:")
    out.append(f"  trades needed         {need:,}")
    out.append(f"  have                  {e.n:,}")
    if e.clusters is not None:
        out.append("  (needed is in trades; with clustered data the binding floor "
                   f"is {e.min_n} clusters, have {e.clusters})")
    if e.n < need:
        out.append(f"  >>> UNDERPOWERED: {need - e.n:,} more needed.")
    else:
        out.append("  adequately powered.")
    return out


def verdict_section(v: Verdict) -> list[str]:
    return ["", "  8. VERDICT", BAR,
            f"  {v.line}",
            f"  rule                  {v.rule.source}"
            + (f"; threshold {v.rule.go_threshold:+.4f}/trade" if v.rule.go_threshold else "")
            + (f"; min {v.rule.min_units} units" if v.rule.min_units is not None else "")
            + f"; drawdown <= {v.rule.max_drawdown_frac:.0%} of bankroll"]


def format_report(a: Analysis, deploy_size: float | None = None) -> str:
    lines = [f"  trades {a.n:,}"]
    for n in a.notes:
        lines.append(f"  note: {n}")
    lines += quality_section(a.quality)
    lines += costs_section(a.costs, a.unit_econ)
    lines += expectancy_section(a.expectancy, a.naive, a.bootstrap, a.claimed_total,
                                a.quality.dependent_without_cluster)
    lines += calibration_section(a.calibration, a.calibration_p, a.calibration_overall)
    if a.split is not None:
        lines += split_section(a.split, a.expectancy.sd)
    if a.drawdown is not None:
        lines += risk_section(a.drawdown, a.ruin, a.bankroll, a.ordered_by_time)
    lines += capacity_section(a.capacity, deploy_size)
    lines += power_section(a.expectancy, a.power_target, a.power_needed)
    lines += verdict_section(a.verdict)
    return "\n".join(lines) + "\n"


def render(trades: list[Trade], split_by: str | None = None,
           unit_econ: UnitEconomics | None = None, buckets: int = 5,
           target: float | None = None, m: ColumnMap | None = None) -> str:
    """0.1-compatible entry point: trades in, text out."""
    a = analyse(trades, m, Options(split_by=split_by, buckets=buckets, target=target,
                                   unit_econ=unit_econ))
    if a is None:
        return "\n  No resolved trades found. Nothing to measure.\n"
    return format_report(a)


# ---- data ------------------------------------------------------------------

def _bucket(b: Bucket, z: float | None, p: float | None) -> dict[str, Any]:
    return {"key": b.key, "predicted": b.predicted, "actual": b.actual, "n": b.n,
            "pnl": b.total_pnl, "gap": b.gap, "z": z, "p_adjusted": p}


def to_dict(a: Analysis, plan_hash: str | None = None) -> dict[str, Any]:
    """The same analysis as JSON-ready data. Field names match the text."""
    e, q, c = a.expectancy, a.quality, a.costs
    d: dict[str, Any] = {
        "edgecheck": VERSION,
        "plan_hash": plan_hash,
        "trades": a.n,
        "quality": {
            "timed": q.timed, "span_seconds": q.span_seconds,
            "median_spacing": q.median_spacing, "gaps": len(q.gaps),
            "gap_frac": q.gap_frac, "duplicate_keys": q.duplicate_keys,
            "shared_minute_frac": q.shared_minute_frac, "clusters": q.clusters,
            "dependent_without_cluster": q.dependent_without_cluster,
            "findings": q.findings,
        },
        "costs": {
            "gross": c.gross, "costs": c.costs, "net": c.net,
            "cost_ratio": c.cost_ratio,
            "structurally_unprofitable": c.structurally_unprofitable,
            "verdict": c.verdict,
        },
        "expectancy": {
            "n": e.n, "clusters": e.clusters, "total": e.total, "mean": e.mean,
            "sd": e.sd, "se": e.se, "lo": e.lo, "hi": e.hi,
            "bootstrap": list(a.bootstrap) if a.bootstrap else None,
            "naive": {"lo": a.naive.lo, "hi": a.naive.hi} if a.naive else None,
            "underpowered": e.underpowered, "distinguishable": e.distinguishable,
            "verdict": e.verdict,
            "claimed_total": a.claimed_total,
            "realisation": (e.total / a.claimed_total
                            if a.claimed_total and abs(a.claimed_total) > 1e-9 else None),
        },
        "calibration": {
            "buckets": [_bucket(b, b.z, p) for b, p in zip(a.calibration, a.calibration_p, strict=True)],
            "overall": ({"predicted": a.calibration_overall[0], "actual": a.calibration_overall[1],
                         "n": a.calibration_overall[2]} if a.calibration_overall else None),
        },
        "split": ({"column": a.split.column,
                   "buckets": [_bucket(b, z, p) for b, z, p in
                               zip(a.split.buckets, a.split.z, a.split.p_adjusted, strict=True)]}
                  if a.split else None),
        "risk": ({"final": a.drawdown.final, "peak": a.drawdown.peak,
                  "max_drawdown": a.drawdown.max_drawdown,
                  "max_drawdown_at": a.drawdown.max_drawdown_at,
                  "longest_underwater": a.drawdown.longest_underwater,
                  "underwater_now": a.drawdown.underwater_now,
                  "bankroll": a.bankroll,
                  "drawdown_frac": (a.drawdown.drawdown_frac(a.bankroll)
                                    if a.bankroll else None),
                  "ruin": ({"probability": a.ruin.probability, "horizon": a.ruin.horizon,
                            "reps": a.ruin.reps} if a.ruin else None),
                  "ordered_by_time": a.ordered_by_time}
                 if a.drawdown else None),
        "capacity": ({"n": a.capacity.n, "tested_size": a.capacity.tested_size,
                      "deploy_size": a.capacity.deploy_size,
                      "fillable_frac": a.capacity.fillable_frac,
                      "limited": a.capacity.limited, "limited_frac": a.capacity.limited_frac,
                      "pnl_tested": a.capacity.pnl_tested,
                      "pnl_at_deploy": a.capacity.pnl_at_deploy,
                      "scale": a.capacity.scale, "linear_scale": a.capacity.linear_scale,
                      "capped": a.capacity.capped}
                     if a.capacity else None),
        "power": {"target": a.power_target, "needed": a.power_needed, "have": e.n},
        "verdict": {"status": a.verdict.status.value, "exit_code": a.verdict.status.exit_code,
                    "reasons": a.verdict.reasons,
                    "rule": {"source": a.verdict.rule.source,
                             "min_units": a.verdict.rule.min_units,
                             "go_threshold": a.verdict.rule.go_threshold,
                             "max_drawdown_frac": a.verdict.rule.max_drawdown_frac}},
        "notes": a.notes,
    }
    return d


__all__ = ["Analysis", "Options", "SplitResult", "analyse", "format_report", "render",
           "to_dict", "SIGNIFICANT_Z"]
