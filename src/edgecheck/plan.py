"""Freeze the protocol before the data arrives.

The cleanest way to fool yourself is to pick the threshold after seeing the
number. Pre-registration is the discipline of writing the threshold down
first; a plan file is that discipline made mechanical. `plan` writes the
target, the clustering, the minimum evidence and the go-threshold to a file
with a hash over all of it. `check` reads the file, refuses to render a
verdict until the minimum evidence is met, scores against the frozen
thresholds, and prints the hash so a reader can confirm nothing moved.

The hash is not security. Anyone can edit the file and rehash it. It is a
tamper-evident seal for the honest: a report that names hash `a1b2c3` can be
checked against a plan file committed on a date, and the diff is the story.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stats import MIN_CLUSTERS_FOR_VERDICT, MIN_N_FOR_VERDICT
from .verdict import MAX_DRAWDOWN_FRAC, Rule

PLAN_VERSION = 1


@dataclass(frozen=True)
class Plan:
    target: float                          # per-trade edge the run is sized to detect
    cluster: str | None = None             # column, or "auto"
    min_units: int | None = None           # trades, or clusters when clustered
    go_threshold: float = 0.0              # mean per trade required for GO
    max_drawdown_frac: float = MAX_DRAWDOWN_FRAC
    bankroll: float | None = None
    columns: dict[str, str] = field(default_factory=dict)
    created: str = ""
    note: str = ""

    @property
    def effective_min_units(self) -> int:
        if self.min_units is not None:
            return self.min_units
        return MIN_CLUSTERS_FOR_VERDICT if self.cluster else MIN_N_FOR_VERDICT

    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d["edgecheck_plan"] = PLAN_VERSION
        return d

    @property
    def sha256(self) -> str:
        canon = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]

    def rule(self) -> Rule:
        return Rule(
            min_units=self.effective_min_units,
            go_threshold=self.go_threshold,
            max_drawdown_frac=self.max_drawdown_frac,
            source=f"plan {self.short_hash}",
        )


class PlanError(ValueError):
    pass


def new_plan(
    target: float,
    cluster: str | None = None,
    min_units: int | None = None,
    go_threshold: float = 0.0,
    max_drawdown_frac: float = MAX_DRAWDOWN_FRAC,
    bankroll: float | None = None,
    columns: dict[str, str] | None = None,
    note: str = "",
) -> Plan:
    return Plan(
        target=target, cluster=cluster, min_units=min_units,
        go_threshold=go_threshold, max_drawdown_frac=max_drawdown_frac,
        bankroll=bankroll, columns=dict(columns or {}),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note=note,
    )


def write_plan(plan: Plan, path: Path) -> str:
    """Write the plan with its hash. Returns the hash."""
    doc = plan.body()
    doc["sha256"] = plan.sha256
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return plan.sha256


def load_plan(path: Path) -> Plan:
    """Read a plan and verify its seal. A plan whose hash does not match its
    contents is refused: that is the one thing the file exists to catch."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("edgecheck_plan") != PLAN_VERSION:
        raise PlanError(f"{path} is not an edgecheck plan (version {PLAN_VERSION})")
    claimed = doc.pop("sha256", None)
    doc.pop("edgecheck_plan")
    try:
        plan = Plan(**doc)
    except TypeError as exc:
        raise PlanError(f"{path}: unexpected field: {exc}") from exc
    if claimed != plan.sha256:
        raise PlanError(
            f"{path}: sha256 does not match its contents -- the plan was edited "
            f"after it was written (file says {str(claimed)[:12]}, contents hash to "
            f"{plan.short_hash})"
        )
    return plan
