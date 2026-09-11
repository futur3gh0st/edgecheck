"""The guards are the product. These pin them down."""

from __future__ import annotations

import random

import pytest

from edgecheck.stats import (
    MIN_N_FOR_VERDICT,
    Z95,
    bucket_by,
    expectancy,
    mean_sd,
    required_n,
    t95,
)


def test_three_lucky_wins_get_no_verdict():
    """The failure this tool exists to prevent: a few wins on high-probability
    bets give a tiny sample sd, a narrow interval, and false confidence."""
    e = expectancy([7.0, 7.1, 6.9])
    assert e.underpowered
    assert not e.distinguishable
    assert "NO VERDICT" in e.verdict


def test_a_real_edge_over_many_trades_is_reported():
    rng = random.Random(7)
    e = expectancy([rng.gauss(5.0, 1.0) for _ in range(200)])
    assert e.distinguishable
    assert e.lo > 0


def test_pure_noise_is_not_distinguishable():
    rng = random.Random(11)
    e = expectancy([rng.gauss(0.0, 10.0) for _ in range(300)])
    assert not e.distinguishable
    assert "NOT distinguishable" in e.verdict


def test_t_critical_is_wider_than_normal_at_small_n():
    """Using 1.96 at n=3 understates the interval by more than 2x."""
    assert t95(2) == pytest.approx(4.303, abs=1e-3)
    assert t95(2) > Z95 * 2
    assert t95(500) == pytest.approx(Z95, abs=0.03)
    assert t95(0) == float("inf")


def test_interval_uses_t_not_normal():
    e = expectancy([7.0, 7.1, 6.9])
    m, sd = mean_sd([7.0, 7.1, 6.9])
    se = sd / (3 ** 0.5)
    assert e.lo == pytest.approx(m - 4.303 * se, abs=1e-6)


def test_required_n_never_returns_a_dishonest_floor():
    """sd estimated from three similar numbers will happily imply 'n=1'."""
    assert required_n(5.0, 0.01) == MIN_N_FOR_VERDICT
    assert required_n(1.0, 0.0) is None
    assert required_n(0.0, 1.0) is None
    big = required_n(0.05, 0.42)
    assert big is not None and big > 500


def test_empty_input_does_not_explode():
    e = expectancy([])
    assert e.n == 0 and not e.distinguishable


def test_buckets_are_equal_count_not_equal_width():
    """A strategy trading only 92-98% would put everything in one fixed-width
    bin and show nothing."""
    rows = [(0.90 + i * 0.0002, 0.9, 1.0, 1.0) for i in range(100)]
    bs = bucket_by(rows, n_buckets=5)
    assert len(bs) == 5
    assert {b.n for b in bs} == {20}


def test_a_short_tail_folds_rather_than_standing_alone():
    """A bucket of two must not be reported as if it meant something."""
    rows = [(float(i), 0.9, 1.0, 1.0) for i in range(22)]
    bs = bucket_by(rows, n_buckets=5, min_rows=5)
    assert all(b.n >= 5 for b in bs)
    assert sum(b.n for b in bs) == 22


def test_gap_sign_means_overconfident():
    rows = [(1.0, 0.95, 0.0, -1.0)] * 10
    b = bucket_by(rows, n_buckets=1)[0]
    assert b.gap < 0          # predicted 95%, won 0% -> overconfident


@pytest.mark.parametrize("n", [1, 3, 7, 12, 22, 47, 100, 451])
@pytest.mark.parametrize("nb", [1, 3, 5, 10])
def test_every_row_lands_in_exactly_one_bucket(n, nb):
    """Bucketing must conserve rows. An earlier version returned an empty table
    whenever n/buckets fell below the minimum, which reads as 'no data'."""
    rows = [(float(i), 0.9, 1.0, 1.0) for i in range(n)]
    bs = bucket_by(rows, n_buckets=nb)
    assert sum(b.n for b in bs) == n
    assert bs, "a non-empty input must always yield at least one bucket"
