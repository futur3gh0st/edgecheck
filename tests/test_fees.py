"""Cost decomposition -- the shortest path to 'stop, this cannot work'."""

from __future__ import annotations

import pytest

from edgecheck.fees import decompose, unit_economics


def test_gross_is_recovered_not_trusted():
    """net + costs must reconcile to gross, always."""
    cb = decompose([1.0, -2.0, 3.0], [0.5, 0.5, 0.5])
    assert cb.net == pytest.approx(2.0)
    assert cb.costs == pytest.approx(1.5)
    assert cb.gross == pytest.approx(3.5)
    assert cb.gross - cb.costs == pytest.approx(cb.net)


def test_costs_are_absolute_however_they_are_signed():
    """Some systems write fees negative, some positive. Both mean money out."""
    a = decompose([1.0, 1.0], [0.25, 0.25])
    b = decompose([1.0, 1.0], [-0.25, -0.25])
    assert a.costs == b.costs == pytest.approx(0.5)


def test_costs_exceeding_edge_is_called_unfixable():
    """The case that ends a project: a real edge, a larger toll."""
    cb = decompose([-0.5] * 100, [1.0] * 100)      # gross +50, costs 100
    assert cb.gross == pytest.approx(50.0)
    assert cb.cost_ratio == pytest.approx(2.0)
    assert cb.structurally_unprofitable
    assert "COSTS EXCEED EDGE" in cb.verdict
    assert "not fixable" in cb.verdict


def test_no_gross_edge_is_distinguished_from_costs_eating_one():
    """Losing before costs is a different diagnosis from losing because of them."""
    cb = decompose([-2.0] * 50, [0.1] * 50)
    assert cb.gross < 0
    assert cb.cost_ratio is None           # ratio is meaningless with no edge
    assert cb.structurally_unprofitable
    assert "NO GROSS EDGE" in cb.verdict


def test_a_healthy_strategy_is_not_flagged():
    cb = decompose([10.0] * 50, [1.0] * 50)
    assert not cb.structurally_unprofitable
    assert cb.cost_ratio == pytest.approx(50 / 550)
    assert "COSTS EXCEED" not in cb.verdict


def test_unit_economics_expose_the_multiple():
    """When the toll is a multiple of the edge, that ratio is the whole story."""
    ue = unit_economics(win_rate=0.70, breakeven=0.68, cost_per_unit=0.04)
    assert ue.edge == pytest.approx(0.02)
    assert ue.cost_multiple == pytest.approx(2.0, abs=0.01)
    assert ue.net < 0
    assert "2.0x the edge" in ue.summary()


def test_unit_economics_with_no_edge_does_not_divide_by_zero():
    ue = unit_economics(win_rate=0.84, breakeven=0.84, cost_per_unit=0.01)
    assert ue.cost_multiple is None
    assert "edge is ~0" in ue.summary()


def test_empty_decomposition_is_safe():
    cb = decompose([], [])
    assert cb.n == 0
    assert cb.gross_per_trade == 0.0 and cb.cost_per_trade == 0.0
