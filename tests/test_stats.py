"""The guards are the product. These pin them down."""

from __future__ import annotations

import random

import pytest

from edgecheck.stats import (
    MIN_CLUSTERS_FOR_VERDICT,
    MIN_N_FOR_VERDICT,
    SIGNIFICANT_Z,
    Z95,
    Bucket,
    binomial_z,
    block_bootstrap,
    bucket_by,
    expectancy,
    holm,
    mean_sd,
    required_n,
    sum_z,
    t95,
    z_to_p,
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


def test_a_small_gap_on_a_small_bucket_is_not_significant():
    """86.4% predicted, 83.8% won, n=80: -2.7 points but z=-0.68. Calling that
    'overconfident' is the overclaim the tool exists to refuse."""
    b = Bucket(key=0.864, predicted=0.864, actual=0.838, n=80, total_pnl=0.0)
    assert b.z == pytest.approx(-0.68, abs=0.01)
    assert not b.significant


def test_the_same_gap_on_a_large_bucket_is_significant():
    b = Bucket(key=0.864, predicted=0.864, actual=0.838, n=2000, total_pnl=0.0)
    assert b.z is not None and b.z < -SIGNIFICANT_Z
    assert b.significant


def test_binomial_z_refuses_a_degenerate_null():
    assert binomial_z(1.0, 0.9, 50) is None
    assert binomial_z(0.0, 0.1, 50) is None
    assert binomial_z(0.5, 0.6, 0) is None


def test_sum_z_scales_with_root_n():
    """A bucket total of -38.71 over 80 trades with per-trade sd 5.44 is under
    one standard error from zero, not a concentrated loss."""
    assert sum_z(-38.71, 5.4358, 80) == pytest.approx(-0.80, abs=0.01)
    assert sum_z(-38.71, 5.4358, 8000) == pytest.approx(-0.08, abs=0.01)
    assert sum_z(1.0, 0.0, 10) is None


# ---- cluster-robust inference ---------------------------------------------

def _clustered_noise(rng, groups=25, per=40, between=1.0, within=0.5):
    """True mean zero. Each cluster shares a draw; trades add their own noise.
    The naive SE sees 1,000 independent trades; there are really 25 outcomes."""
    pnl, cl = [], []
    for g in range(groups):
        shock = rng.gauss(0.0, between)
        for _ in range(per):
            pnl.append(shock + rng.gauss(0.0, within))
            cl.append(f"g{g}")
    return pnl, cl


def test_naive_interval_undercovers_on_clustered_noise():
    """The hole 0.1 had. On dependent trades the per-trade interval claims 95%
    and delivers far less. This test documents the failure the fix removes."""
    rng = random.Random(3)
    covered = 0
    reps = 200
    for _ in range(reps):
        pnl, _ = _clustered_noise(rng)
        e = expectancy(pnl)
        covered += e.lo <= 0.0 <= e.hi
    assert covered / reps < 0.75


def test_cluster_interval_covers_at_the_stated_rate():
    """Same data, cluster-robust: about 95 in 100 intervals contain the truth.
    Tolerance is wide enough not to be flaky and tight enough to catch a
    formula error -- CR1 without the G/(G-1) factor lands near 0.90."""
    rng = random.Random(3)
    covered = 0
    reps = 300
    for _ in range(reps):
        pnl, cl = _clustered_noise(rng)
        e = expectancy(pnl, cl)
        covered += e.lo <= 0.0 <= e.hi
    assert 0.91 <= covered / reps <= 0.99


def test_cluster_interval_counts_clusters_not_trades():
    pnl, cl = _clustered_noise(random.Random(1))
    e = expectancy(pnl, cl)
    assert e.n == 1000 and e.clusters == 25
    assert e.units == 25 and e.unit_name == "clusters"
    assert e.min_n == MIN_CLUSTERS_FOR_VERDICT
    assert not e.underpowered


def test_too_few_clusters_is_no_verdict_however_many_trades():
    rng = random.Random(2)
    pnl, cl = _clustered_noise(rng, groups=5, per=500)
    e = expectancy(pnl, cl)
    assert e.n == 2500 and e.clusters == 5
    assert e.underpowered
    assert "5 clusters, need 21" in e.verdict


def test_a_trade_without_a_cluster_is_its_own_cluster():
    e = expectancy([1.0, 2.0, 3.0, 4.0], [None, None, "a", "a"])
    assert e.clusters == 3


def test_one_cluster_has_no_interval():
    e = expectancy([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert e.se == 0.0 and e.lo == e.hi == e.mean


def test_misaligned_clusters_are_refused():
    with pytest.raises(ValueError):
        expectancy([1.0, 2.0], ["a"])


def test_min_n_can_be_overridden_by_a_plan():
    e = expectancy([1.0] * 40, min_n=50)
    assert e.underpowered
    assert "need 50" in e.verdict


def test_cluster_interval_matches_t_when_every_trade_is_its_own_cluster():
    """With singleton clusters CR1 reduces to the ordinary variance times
    n/(n-1) -- within a whisker of the t interval, not wildly different."""
    rng = random.Random(9)
    pnl = [rng.gauss(0.3, 2.0) for _ in range(200)]
    naive = expectancy(pnl)
    solo = expectancy(pnl, [str(i) for i in range(200)])
    assert solo.se == pytest.approx(naive.se, rel=0.01)


# ---- bootstrap --------------------------------------------------------------

def test_bootstrap_agrees_with_t_on_iid_data():
    rng = random.Random(5)
    pnl = [rng.gauss(1.0, 3.0) for _ in range(400)]
    e = expectancy(pnl)
    b = block_bootstrap(pnl, reps=2000)
    assert b is not None
    lo, hi = b
    assert lo == pytest.approx(e.lo, abs=0.1)
    assert hi == pytest.approx(e.hi, abs=0.1)


def test_block_bootstrap_is_wider_than_naive_on_clustered_data():
    pnl, cl = _clustered_noise(random.Random(4))
    naive = block_bootstrap(pnl, reps=1000)
    block = block_bootstrap(pnl, cl, reps=1000)
    assert naive is not None and block is not None
    assert (block[1] - block[0]) > 2 * (naive[1] - naive[0])


def test_bootstrap_is_deterministic():
    pnl = [float(i % 7) - 3 for i in range(60)]
    assert block_bootstrap(pnl) == block_bootstrap(pnl)


def test_bootstrap_needs_two_blocks():
    assert block_bootstrap([1.0]) is None
    assert block_bootstrap([1.0, 2.0, 3.0], ["a", "a", "a"]) is None


# ---- multiple comparisons ---------------------------------------------------

def test_holm_leaves_a_single_test_alone():
    assert holm([0.03]) == [0.03]


def test_holm_penalises_the_smallest_p_by_the_family_size():
    adj = holm([0.01, 0.20, 0.04, 0.50, 0.30])
    assert adj[0] == pytest.approx(0.05)          # 5 * 0.01
    assert adj[2] == pytest.approx(0.16)          # 4 * 0.04
    assert all(a <= 1.0 for a in adj)


def test_holm_is_monotone_in_the_input_order_of_p():
    adj = holm([0.04, 0.041, 0.042, 0.043, 0.044])
    ranked = sorted(zip([0.04, 0.041, 0.042, 0.043, 0.044], adj, strict=True))
    assert all(a <= b for (_, a), (_, b) in zip(ranked, ranked[1:], strict=False))


def test_one_lucky_bucket_in_five_is_not_significant_after_holm():
    """z = 2.1 on its own clears |z| >= 2. As the worst of five it does not."""
    p = z_to_p(2.1)
    assert p < 0.05
    assert holm([p, 0.4, 0.5, 0.6, 0.7])[0] > 0.05


def test_z_to_p_is_two_sided():
    assert z_to_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert z_to_p(-1.96) == z_to_p(1.96)


def test_required_n_uses_t_and_so_asks_for_slightly_more():
    """Normal arithmetic says ceil(7.85 * 4) = 32; t at df~31 nudges it up."""
    n = required_n(effect=1.0, sd=2.0)
    assert n is not None
    assert 32 <= n <= 35
