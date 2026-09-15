"""One answer, and the rule that produced it."""

from __future__ import annotations

import random

from edgecheck.fees import decompose
from edgecheck.loaders import ColumnMap, Trade
from edgecheck.quality import assess
from edgecheck.risk import drawdown
from edgecheck.stats import expectancy
from edgecheck.verdict import Rule, Status, decide


def _winner(n=200, seed=1):
    rng = random.Random(seed)
    return [rng.gauss(2.0, 3.0) for _ in range(n)]


def _noise(n=200, seed=4):          # seed 4: net +45 over 200, interval straddles zero
    rng = random.Random(seed)
    return [rng.gauss(0.0, 3.0) for _ in range(n)]


def _costs(pnl, fee=0.1):
    return decompose(pnl, [fee] * len(pnl))


def test_a_real_edge_is_go():
    pnl = _winner()
    v = decide(expectancy(pnl), _costs(pnl))
    assert v.status is Status.GO
    assert v.status.exit_code == 0
    assert "above zero" in v.line


def test_too_few_trades_is_no_verdict_before_anything_else():
    """Even a strategy that loses on costs gets NO_VERDICT at n=5: the point
    is that nothing is known yet, including that."""
    pnl = [-5.0] * 5
    v = decide(expectancy(pnl), _costs(pnl, fee=10.0))
    assert v.status is Status.NO_VERDICT
    assert v.status.exit_code == 3
    assert "need 30" in v.line


def test_costs_exceeding_edge_is_no_go_with_the_reason_named():
    pnl = [-0.5] * 50                      # net -0.5 after a fee of 1.0: gross +0.5
    v = decide(expectancy(pnl), _costs(pnl, fee=1.0))
    assert v.status is Status.NO_GO
    assert "costs exceed edge" in v.line


def test_noise_with_enough_data_is_no_go():
    pnl = _noise()
    v = decide(expectancy(pnl), _costs(pnl, fee=0.05))  # costs small; net is noise
    assert v.status is Status.NO_GO
    assert "cannot distinguish" in v.line


def test_an_interval_below_zero_is_no_go():
    pnl = [-2.0 + 0.1 * (i % 3) for i in range(60)]
    v = decide(expectancy(pnl), _costs(pnl))
    assert v.status is Status.NO_GO
    assert "below zero" in v.line


def test_dependent_trades_without_a_cluster_cap_the_verdict():
    """A hundred trades in the same minute would be GO on the naive interval.
    The quality pass says they are one piece of evidence; the verdict listens."""
    pnl = _winner(100)
    ts = [Trade(pnl=x, time=0.0) for x in pnl]
    q = assess(ts, ColumnMap(time="ts"))
    v = decide(expectancy(pnl), _costs(pnl), quality=q)
    assert v.status is Status.NO_VERDICT
    assert "not independent" in v.line


def test_a_survivable_edge_with_an_unsurvivable_drawdown_is_no_go():
    pnl = [-30.0] * 3 + [3.0] * 200         # interval above zero; path opens with -90
    e = expectancy(pnl)
    assert e.lo > 0
    v = decide(e, _costs(pnl), dd=drawdown(pnl), bankroll=100.0)
    assert v.status is Status.NO_GO
    assert "drawdown 90%" in v.line


def test_the_same_drawdown_with_a_bigger_bankroll_is_go():
    pnl = [-30.0] * 3 + [3.0] * 200
    v = decide(expectancy(pnl), _costs(pnl), dd=drawdown(pnl), bankroll=1000.0)
    assert v.status is Status.GO


def test_a_plan_can_raise_the_evidence_floor():
    pnl = _winner(40)
    assert decide(expectancy(pnl), _costs(pnl)).status is Status.GO
    v = decide(expectancy(pnl), _costs(pnl), rule=Rule(min_units=100, source="plan abc"))
    assert v.status is Status.NO_VERDICT
    assert "need 100" in v.line
    assert v.rule.source == "plan abc"


def test_a_plan_go_threshold_turns_a_small_real_edge_into_no_go():
    pnl = _winner()                               # mean ~2
    v = decide(expectancy(pnl), _costs(pnl), rule=Rule(go_threshold=5.0))
    assert v.status is Status.NO_GO
    assert "below threshold" in v.line
