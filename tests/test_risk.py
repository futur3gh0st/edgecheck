"""Would you have survived the path to the mean?"""

from __future__ import annotations

import random

import pytest

from edgecheck.risk import drawdown, ruin_probability


def test_drawdown_on_a_hand_built_path():
    #  equity: 1, 3, 2, 0, 4, 1, 2
    #  peak 3 -> trough 0 at index 3 is a drawdown of 3; peak 4 -> trough 1 at
    #  index 5 ties it and does not replace it, so the first is reported.
    d = drawdown([1.0, 2.0, -1.0, -2.0, 4.0, -3.0, 1.0])
    assert d.final == 2.0
    assert d.peak == 4.0
    assert d.max_drawdown == 3.0
    assert d.max_drawdown_at == 3
    assert d.longest_underwater == 2          # trades 2-3 and trades 5-6
    assert d.underwater_now


def test_a_monotone_winner_has_no_drawdown():
    d = drawdown([1.0] * 10)
    assert d.max_drawdown == 0.0 and d.longest_underwater == 0 and not d.underwater_now


def test_drawdown_is_measured_from_the_starting_point_too():
    """A strategy that opens with losses is under water from trade one; the
    peak is zero, not the first observation."""
    d = drawdown([-2.0, -3.0, 1.0])
    assert d.max_drawdown == 5.0
    assert d.peak == 0.0


def test_drawdown_frac_needs_a_positive_bankroll():
    d = drawdown([-5.0])
    assert d.drawdown_frac(100.0) == pytest.approx(0.05)
    assert d.drawdown_frac(0.0) is None


def test_empty_path_is_safe():
    d = drawdown([])
    assert d.n == 0 and d.max_drawdown == 0.0


def test_ruin_is_certain_when_the_bankroll_is_smaller_than_one_loss():
    r = ruin_probability([-10.0, -10.0, -10.0, -10.0], bankroll=5.0, reps=200)
    assert r is not None
    assert r.probability == 1.0


def test_ruin_is_impossible_for_a_strategy_that_never_loses():
    r = ruin_probability([1.0, 2.0, 3.0, 4.0], bankroll=1.0, reps=200)
    assert r is not None
    assert r.probability == 0.0


def test_ruin_rises_as_the_bankroll_shrinks():
    rng = random.Random(0)
    pnl = [rng.gauss(0.2, 5.0) for _ in range(300)]
    small = ruin_probability(pnl, bankroll=20.0, reps=1000)
    large = ruin_probability(pnl, bankroll=200.0, reps=1000)
    assert small is not None and large is not None
    assert small.probability > large.probability


def test_ruin_resamples_clusters_intact():
    """Two clusters: one all wins, one all losses. Resampling trades would mix
    them and soften the losing streak; resampling clusters cannot."""
    pnl = [5.0] * 10 + [-5.0] * 10
    cl = ["w"] * 10 + ["l"] * 10
    r = ruin_probability(pnl, bankroll=40.0, clusters=cl, horizon=10, reps=500)
    assert r is not None
    # a drawn losing cluster loses 50 in 10 trades > 40 bankroll: ruin iff first block is 'l'
    assert r.probability == pytest.approx(0.5, abs=0.08)


def test_ruin_is_deterministic():
    pnl = [float(i % 5) - 2.5 for i in range(50)]
    a = ruin_probability(pnl, bankroll=10.0, reps=300)
    b = ruin_probability(pnl, bankroll=10.0, reps=300)
    assert a is not None and b is not None and a.ruined == b.ruined


def test_ruin_needs_data_and_a_bankroll():
    assert ruin_probability([1.0], bankroll=10.0) is None
    assert ruin_probability([1.0, 2.0], bankroll=0.0) is None
    assert ruin_probability([1.0, 2.0], bankroll=5.0, clusters=["a", "a"]) is None
