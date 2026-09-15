"""Is the data fit to be measured?"""

from __future__ import annotations

from edgecheck.loaders import ColumnMap, Trade
from edgecheck.quality import GAP_MULTIPLE, assess


def _timed(times: list[float], **kw) -> list[Trade]:
    return [Trade(pnl=1.0, time=t, **kw) for t in times]


def test_a_clean_evenly_spaced_log_has_no_findings():
    q = assess(_timed([float(i * 60) for i in range(50)]), ColumnMap(time="ts"))
    assert q.clean
    assert q.median_spacing == 60.0
    assert q.gaps == []


def test_a_recorder_outage_shows_up_as_a_gap():
    """Fifty trades a minute apart, then nothing for a day, then fifty more.
    The day is the finding; the fifty-minute halves are the strategy."""
    a = [float(i * 60) for i in range(50)]
    b = [a[-1] + 86400 + i * 60 for i in range(50)]
    q = assess(_timed(a + b), ColumnMap(time="ts"))
    assert len(q.gaps) == 1
    assert q.gaps[0].seconds == 86400.0
    assert q.gap_frac > 0.9
    assert any("was the recorder running" in f for f in q.findings)


def test_a_gap_just_under_the_multiple_is_not_reported():
    a = [float(i * 60) for i in range(20)]
    b = [a[-1] + (GAP_MULTIPLE - 0.5) * 60 + i * 60 for i in range(20)]
    assert assess(_timed(a + b), ColumnMap(time="ts")).gaps == []


def test_duplicate_keys_are_counted():
    ts = [Trade(pnl=1.0, raw={"id": k}) for k in ("A", "B", "A", "A", "C")]
    q = assess(ts, ColumnMap(key="id"))
    assert q.duplicate_keys == 2
    assert any("duplicate" in f for f in q.findings)


def test_shared_minutes_without_a_cluster_is_called_out():
    """Ten trades in the same minute: they share an outcome. With no cluster
    named the naive interval is an overclaim, and the report must say so."""
    q = assess(_timed([0.0] * 10 + [600.0]), ColumnMap(time="ts"))
    assert q.dependent_without_cluster
    assert any("not independent" in f for f in q.findings)


def test_shared_minutes_with_a_cluster_is_fine():
    ts = _timed([0.0] * 10 + [600.0], cluster="d1")
    q = assess(ts, ColumnMap(time="ts", cluster="day"))
    assert not q.dependent_without_cluster
    assert q.clusters == 1


def test_a_lopsided_side_split_hints_at_the_p_yes_trap():
    rows = [Trade(pnl=1.0, raw={"side": "yes"}) for _ in range(19)]
    rows.append(Trade(pnl=1.0, raw={"side": "no"}))
    q = assess(rows, ColumnMap(side="side"))
    assert q.lopsided
    assert any("--predicted-is-yes" in f for f in q.findings)


def test_a_lopsided_split_on_a_tiny_sample_is_not_called():
    rows = [Trade(pnl=1.0, raw={"side": "yes"}) for _ in range(9)]
    rows.append(Trade(pnl=1.0, raw={"side": "no"}))
    assert not assess(rows, ColumnMap(side="side")).lopsided


def test_missing_predictions_and_outcomes_are_counted_only_when_mapped():
    ts = [Trade(pnl=1.0, predicted=0.7, won=True), Trade(pnl=1.0), Trade(pnl=1.0, predicted=0.6)]
    q = assess(ts, ColumnMap(predicted="p", won="w"))
    assert (q.missing_predicted, q.missing_won) == (1, 2)
    assert assess(ts, ColumnMap()).missing_predicted == 0


def test_unreadable_times_are_reported():
    ts = _timed([0.0, 60.0, 120.0]) + [Trade(pnl=1.0)]
    q = assess(ts, ColumnMap(time="ts"))
    assert q.missing_time == 1
    assert any("unreadable time" in f for f in q.findings)


def test_fewer_than_two_times_yields_no_span():
    q = assess(_timed([5.0]), ColumnMap(time="ts"))
    assert q.span_seconds is None and q.median_spacing is None and q.gaps == []


def test_empty_input_is_clean():
    assert assess([], ColumnMap()).clean
