"""One answer, and the rule that produced it.

Five sections each say something true. The reader still has to decide, and
the deciding is where the fooling happens -- a costs line that reads "82%"
is easy to wave through when the expectancy line underneath says
"DISTINGUISHABLE". So the report ends with one line, the rule is printed
beside it, and the exit code carries it, so a script cannot wave it through
either.

The rule is deliberately conservative and deliberately explicit. Every
condition that can produce NO_GO is listed in `reasons`, and NO_VERDICT names
what is missing. A plan file can replace the thresholds; it cannot remove a
condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .fees import CostBreakdown
from .quality import DataQuality
from .risk import Drawdown
from .stats import Expectancy


class Status(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    NO_VERDICT = "NO_VERDICT"

    @property
    def exit_code(self) -> int:
        return {Status.GO: 0, Status.NO_GO: 1, Status.NO_VERDICT: 3}[self]


# Default share of a stated bankroll that the max drawdown may consume before
# the strategy is called unrunnable regardless of its mean.
MAX_DRAWDOWN_FRAC = 0.5


@dataclass(frozen=True)
class Rule:
    """The thresholds a verdict is judged against. Defaults here; a plan
    overrides them and is named on the report."""

    min_units: int | None = None          # None: whatever expectancy() used
    go_threshold: float = 0.0             # mean per trade must reach this
    max_drawdown_frac: float = MAX_DRAWDOWN_FRAC
    source: str = "default"               # "default" or "plan <hash>"


@dataclass(frozen=True)
class Verdict:
    status: Status
    rule: Rule
    reasons: list[str] = field(default_factory=list)

    @property
    def line(self) -> str:
        return f"{self.status.value}" + (f" -- {'; '.join(self.reasons)}" if self.reasons else "")


def decide(
    e: Expectancy,
    costs: CostBreakdown,
    quality: DataQuality | None = None,
    dd: Drawdown | None = None,
    bankroll: float | None = None,
    rule: Rule | None = None,
) -> Verdict:
    """Apply the rule, in the order that reaches a decision fastest.

    NO_VERDICT  not enough independent evidence, or the evidence is not
                independent and nobody said how it clusters
    NO_GO       structurally unprofitable, or the interval sits below zero,
                or the mean misses the threshold, or the drawdown would have
                exhausted the stated bankroll
    GO          interval above zero, mean at threshold, drawdown survivable
    NO_GO       enough evidence, and it cannot tell the edge from zero
    """
    r = rule or Rule()
    min_units = r.min_units if r.min_units is not None else e.min_n

    if e.units < min_units:
        return Verdict(Status.NO_VERDICT, r,
                       [f"{e.units} {e.unit_name}, need {min_units}"])
    if quality is not None and quality.dependent_without_cluster:
        return Verdict(Status.NO_VERDICT, r,
                       ["trades are not independent and no --cluster was given"])

    reasons: list[str] = []
    if costs.structurally_unprofitable:
        reasons.append(costs.verdict.split(" -- ")[0].lower())
    if e.se > 0 and e.hi < 0:
        reasons.append("interval entirely below zero")
    if dd is not None and bankroll is not None and bankroll > 0:
        frac = dd.drawdown_frac(bankroll)
        if frac is not None and frac > r.max_drawdown_frac:
            reasons.append(f"max drawdown {frac:.0%} of bankroll exceeds {r.max_drawdown_frac:.0%}")
    if reasons:
        return Verdict(Status.NO_GO, r, reasons)

    if e.se > 0 and e.lo > 0 and e.mean >= r.go_threshold:
        return Verdict(Status.GO, r, [f"interval [{e.lo:+.4f}, {e.hi:+.4f}] above zero"]
                       + ([f"mean {e.mean:+.4f} >= threshold {r.go_threshold:+.4f}"]
                          if r.go_threshold else []))
    if e.se > 0 and e.lo > 0:
        return Verdict(Status.NO_GO, r,
                       [f"mean {e.mean:+.4f} below threshold {r.go_threshold:+.4f}"])
    return Verdict(Status.NO_GO, r, ["cannot distinguish the edge from zero"])
