"""Freeze the protocol before the data arrives."""

from __future__ import annotations

import json

import pytest

from edgecheck.plan import PlanError, load_plan, new_plan, write_plan
from edgecheck.stats import MIN_CLUSTERS_FOR_VERDICT, MIN_N_FOR_VERDICT


def test_a_plan_round_trips_with_its_hash(tmp_path):
    p = new_plan(target=0.05, cluster="date", min_units=21, go_threshold=10.0,
                 columns={"pnl": "pnl", "time": "ts"}, note="maker forward test")
    h = write_plan(p, tmp_path / "plan.json")
    back = load_plan(tmp_path / "plan.json")
    assert back == p
    assert back.sha256 == h
    assert len(h) == 64


def test_an_edited_plan_is_refused(tmp_path):
    """The one thing the file exists to catch: a threshold moved after the
    fact. The hash no longer matches, and the reason names both hashes."""
    p = new_plan(target=0.05, go_threshold=10.0)
    write_plan(p, tmp_path / "plan.json")
    doc = json.loads((tmp_path / "plan.json").read_text())
    doc["go_threshold"] = 1.0
    (tmp_path / "plan.json").write_text(json.dumps(doc))
    with pytest.raises(PlanError, match="edited after it was written"):
        load_plan(tmp_path / "plan.json")


def test_the_hash_covers_every_field():
    a = new_plan(target=0.05, note="x")
    b = new_plan(target=0.05, note="y")
    assert a.sha256 != b.sha256


def test_a_file_that_is_not_a_plan_is_refused(tmp_path):
    (tmp_path / "x.json").write_text('{"hello": 1}')
    with pytest.raises(PlanError, match="not an edgecheck plan"):
        load_plan(tmp_path / "x.json")
    (tmp_path / "y.json").write_text("not json")
    with pytest.raises(PlanError, match="cannot read"):
        load_plan(tmp_path / "y.json")


def test_unknown_fields_are_refused_rather_than_ignored(tmp_path):
    p = new_plan(target=0.05)
    write_plan(p, tmp_path / "plan.json")
    doc = json.loads((tmp_path / "plan.json").read_text())
    doc["min_trades"] = 500                # a typo for min_units would silently do nothing
    (tmp_path / "plan.json").write_text(json.dumps(doc))
    with pytest.raises(PlanError, match="unexpected field"):
        load_plan(tmp_path / "plan.json")


def test_evidence_floor_follows_clustering_unless_set():
    assert new_plan(target=0.1).effective_min_units == MIN_N_FOR_VERDICT
    assert new_plan(target=0.1, cluster="auto").effective_min_units == MIN_CLUSTERS_FOR_VERDICT
    assert new_plan(target=0.1, cluster="auto", min_units=7).effective_min_units == 7


def test_the_rule_names_the_plan():
    p = new_plan(target=0.05, go_threshold=2.0, max_drawdown_frac=0.25)
    r = p.rule()
    assert r.go_threshold == 2.0 and r.max_drawdown_frac == 0.25
    assert r.source == f"plan {p.short_hash}"
