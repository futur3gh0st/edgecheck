"""Does the edge survive the size you would actually run?"""

from __future__ import annotations

import pytest

from edgecheck.capacity import capacity
from edgecheck.loaders import Trade


def _t(pnl, size, depth=None, fill=None):
    return Trade(pnl=pnl, size=size, depth=depth, fill_size=fill)


def test_deploying_into_a_deep_book_scales_linearly():
    ts = [_t(1.0, 10.0, depth=100.0) for _ in range(5)]
    c = capacity(ts, deploy_size=30.0)
    assert c is not None
    assert c.limited == 0
    assert c.fillable_frac == 1.0
    assert c.pnl_at_deploy == pytest.approx(15.0)      # 3x the size, 3x the P&L
    assert c.scale == pytest.approx(c.linear_scale)


def test_deploying_into_a_thin_book_is_capped():
    """The crypto-bot finding: +222 at $2 depth does not become +1,664 at $15.
    Each trade earns p/s per unit and only `depth` units are fillable."""
    ts = [_t(1.0, 2.0, depth=2.0) for _ in range(10)]
    c = capacity(ts, deploy_size=15.0)
    assert c is not None
    assert c.limited == 10 and c.capped
    assert c.pnl_at_deploy == pytest.approx(10.0)      # exactly what filled at $2
    assert c.linear_scale == pytest.approx(7.5)
    assert c.scale == pytest.approx(1.0)


def test_fill_size_is_preferred_over_depth_when_both_exist():
    ts = [_t(2.0, 4.0, depth=100.0, fill=4.0)]
    c = capacity(ts, deploy_size=8.0)
    assert c is not None
    assert c.limited == 1                              # fill of 4 < deploy of 8
    assert c.pnl_at_deploy == pytest.approx(2.0)


def test_a_mix_of_limited_and_unlimited_trades():
    ts = [_t(1.0, 1.0, depth=10.0), _t(1.0, 1.0, depth=2.0)]
    c = capacity(ts, deploy_size=5.0)
    assert c is not None
    assert c.limited == 1 and not c.capped
    assert c.fillable_frac == pytest.approx((1.0 + 0.4) / 2)
    assert c.pnl_at_deploy == pytest.approx(5.0 + 2.0)


def test_trades_without_size_information_are_ignored():
    ts = [_t(1.0, 1.0, depth=5.0), Trade(pnl=9.0), _t(1.0, None, depth=5.0)]
    c = capacity(ts, deploy_size=2.0)
    assert c is not None and c.n == 1


def test_nothing_measurable_returns_none():
    assert capacity([Trade(pnl=1.0)], deploy_size=5.0) is None
    assert capacity([_t(1.0, 1.0, depth=5.0)], deploy_size=0.0) is None


def test_scale_is_none_when_tested_pnl_is_zero():
    c = capacity([_t(0.0, 1.0, depth=5.0)], deploy_size=2.0)
    assert c is not None and c.scale is None
