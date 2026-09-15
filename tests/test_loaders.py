"""Reading whatever shape the data already has."""

from __future__ import annotations

import json

import pytest

from edgecheck.loaders import ColumnMap, load, parse_time, read_rows


def _csv(tmp_path, header, rows, name="t.csv"):
    p = tmp_path / name
    lines = [",".join(header)] + [",".join(str(c) for c in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return p


def test_reads_csv(tmp_path):
    p = _csv(tmp_path, ["pnl", "fee"], [[1.5, 0.1], [-2.0, 0.1]])
    ts = load(p, ColumnMap(pnl="pnl", cost="fee"))
    assert [t.pnl for t in ts] == [1.5, -2.0]
    assert [t.cost for t in ts] == [0.1, 0.1]


def test_reads_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps({"pnl": v}) for v in (1.0, 2.0)) + "\n")
    assert [t.pnl for t in load(p, ColumnMap())] == [1.0, 2.0]


def test_a_partial_final_line_is_tolerated(tmp_path):
    """Live logs get read while they are being appended to."""
    p = tmp_path / "t.jsonl"
    p.write_text('{"pnl": 1.0}\n{"pnl": 2.0}\n{"pnl": 3.0')     # truncated
    assert len(load(p, ColumnMap())) == 2


def test_json_extension_holding_a_list(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"pnl": 1.0}, {"pnl": 2.0}]))
    assert len(load(p, ColumnMap())) == 2


def test_rows_without_a_pnl_are_skipped_not_zeroed(tmp_path):
    """A missing result is not a result of zero."""
    p = _csv(tmp_path, ["pnl"], [[1.0], [""], [3.0]])
    assert [t.pnl for t in load(p, ColumnMap())] == [1.0, 3.0]


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("won", True),
    ("false", False), ("0", False), ("no", False), ("", None),
])
def test_won_accepts_the_spellings_people_actually_use(tmp_path, raw, expected):
    p = _csv(tmp_path, ["pnl", "won"], [[1.0, raw]])
    assert load(p, ColumnMap(won="won"))[0].won is expected


def test_p_yes_is_flipped_for_a_no_side_trade(tmp_path):
    """Storing P(YES) and reading it as P(side taken) scores a no-side trade
    against its own complement. Invisible until a no-side trade appears."""
    p = _csv(tmp_path, ["pnl", "fair", "side"], [[1.0, 0.70, "no"], [1.0, 0.70, "yes"]])
    m = ColumnMap(predicted="fair", side="side", predicted_is_yes=True)
    ts = load(p, m)
    assert ts[0].predicted == pytest.approx(0.30)
    assert ts[1].predicted == pytest.approx(0.70)


def test_p_side_is_left_alone_when_not_flagged(tmp_path):
    p = _csv(tmp_path, ["pnl", "fair", "side"], [[1.0, 0.30, "no"]])
    ts = load(p, ColumnMap(predicted="fair", side="side"))
    assert ts[0].predicted == pytest.approx(0.30)


def test_claimed_edge_scales_by_size(tmp_path):
    p = _csv(tmp_path, ["pnl", "edge", "shares"], [[1.0, 0.05, 200]])
    t = load(p, ColumnMap(claimed="edge", size="shares"))[0]
    assert t.claimed_value == pytest.approx(10.0)


def test_claimed_edge_without_size_is_per_unit(tmp_path):
    p = _csv(tmp_path, ["pnl", "edge"], [[1.0, 0.05]])
    assert load(p, ColumnMap(claimed="edge"))[0].claimed_value == pytest.approx(0.05)


def test_paired_log_joins_entry_to_resolve(tmp_path):
    p = tmp_path / "live.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "open", "id": "A", "fair": 0.8},
        {"kind": "settle", "id": "A", "pnl": 1.5, "won": True},
    ]) + "\n")
    m = ColumnMap(kind="kind", entry_kind="open", resolve_kind="settle",
                  key="id", predicted="fair", won="won")
    ts = load(p, m)
    assert len(ts) == 1
    assert ts[0].pnl == 1.5 and ts[0].predicted == 0.8 and ts[0].won is True


def test_repeated_keys_pair_in_order(tmp_path):
    """Same market traded twice: the second resolve must not rebind to the
    first entry and reuse its claimed edge."""
    p = tmp_path / "live.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "open", "id": "SAME", "fair": 0.6},
        {"kind": "open", "id": "SAME", "fair": 0.7},
        {"kind": "settle", "id": "SAME", "pnl": 1.0},
        {"kind": "settle", "id": "SAME", "pnl": -1.0},
    ]) + "\n")
    m = ColumnMap(kind="kind", entry_kind="open", resolve_kind="settle",
                  key="id", predicted="fair")
    ts = load(p, m)
    assert [t.predicted for t in ts] == [0.6, 0.7]


def test_an_unmatched_entry_is_not_counted(tmp_path):
    """Open positions have no outcome yet and must not be scored."""
    p = tmp_path / "live.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "open", "id": "A", "fair": 0.8},
        {"kind": "open", "id": "B", "fair": 0.9},
        {"kind": "settle", "id": "A", "pnl": 1.0},
    ]) + "\n")
    m = ColumnMap(kind="kind", entry_kind="open", resolve_kind="settle", key="id")
    assert len(load(p, m)) == 1


def test_read_rows_on_an_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert read_rows(p) == []


def test_paired_needs_a_kind_column():
    """key + entry_kind + resolve_kind without kind used to count as paired and
    then look every row up under a None column."""
    m = ColumnMap(key="id", entry_kind="open", resolve_kind="settle")
    assert not m.paired
    assert ColumnMap(key="id", kind="kind", entry_kind="open", resolve_kind="settle").paired


# ---- time, cluster, capacity columns ---------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2026-09-14T15:32:00Z", 1789399920.0),
    ("2026-09-14T15:32:00+00:00", 1789399920.0),
    ("2026-09-14 15:32:00", 1789399920.0),          # naive -> UTC
    ("1789399920", 1789399920.0),                    # epoch seconds
    ("1789399920000", 1789399920.0),                 # epoch milliseconds
    ("not a time", None),
    ("", None),
])
def test_parse_time_reads_the_shapes_logs_actually_use(raw, expected):
    assert parse_time(raw) == expected


def test_trades_come_back_in_entry_order_when_timed(tmp_path):
    """Drawdown and gap detection are meaningless on file order."""
    p = _csv(tmp_path, ["pnl", "ts"], [[3.0, "2026-01-03"], [1.0, "2026-01-01"], [2.0, "2026-01-02"]])
    assert [t.pnl for t in load(p, ColumnMap(time="ts"))] == [1.0, 2.0, 3.0]


def test_unreadable_times_are_kept_last_not_dropped(tmp_path):
    p = _csv(tmp_path, ["pnl", "ts"], [[2.0, "2026-01-02"], [9.0, "junk"], [1.0, "2026-01-01"]])
    ts = load(p, ColumnMap(time="ts"))
    assert [t.pnl for t in ts] == [1.0, 2.0, 9.0]
    assert ts[-1].time is None


def test_auto_cluster_is_the_utc_day(tmp_path):
    p = _csv(tmp_path, ["pnl", "ts"], [
        [1.0, "2026-01-01T23:59:00Z"], [1.0, "2026-01-02T00:01:00Z"], [1.0, "2026-01-02T12:00:00Z"],
    ])
    ts = load(p, ColumnMap(time="ts", cluster="auto"))
    assert [t.cluster for t in ts] == ["2026-01-01", "2026-01-02", "2026-01-02"]


def test_auto_cluster_without_time_is_none(tmp_path):
    p = _csv(tmp_path, ["pnl"], [[1.0]])
    assert load(p, ColumnMap(cluster="auto"))[0].cluster is None


def test_named_cluster_column_is_read_as_text(tmp_path):
    p = _csv(tmp_path, ["pnl", "sess"], [[1.0, 7], [1.0, "7"], [1.0, ""]])
    assert [t.cluster for t in load(p, ColumnMap(cluster="sess"))] == ["7", "7", None]


def test_time_comes_from_the_entry_row_in_a_paired_log(tmp_path):
    """The decision was made at entry; settlement time is when the world
    caught up, and clustering by it would group unrelated decisions."""
    p = tmp_path / "live.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "open", "id": "A", "ts": "2026-01-01T10:00:00Z"},
        {"kind": "settle", "id": "A", "pnl": 1.0, "ts": "2026-01-01T18:00:00Z"},
    ]) + "\n")
    m = ColumnMap(kind="kind", entry_kind="open", resolve_kind="settle", key="id", time="ts")
    assert load(p, m)[0].time == parse_time("2026-01-01T10:00:00Z")


def test_fill_size_and_depth_are_read(tmp_path):
    p = _csv(tmp_path, ["pnl", "size", "filled", "depth"], [[1.0, 15, 2, 2.5]])
    t = load(p, ColumnMap(size="size", fill_size="filled", depth="depth"))[0]
    assert (t.size, t.fill_size, t.depth) == (15.0, 2.0, 2.5)
