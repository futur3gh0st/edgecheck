"""End-to-end: the shipped examples must produce the verdicts they illustrate."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgecheck.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_fee_eaten_example_is_called_unfixable(capsys):
    rc = main([str(EXAMPLES / "fee_eaten.csv"), "--pnl", "pnl", "--cost", "fee",
               "--predicted", "model_p", "--won", "won"])
    out = capsys.readouterr().out
    assert rc == 1                                   # NO_GO travels in the exit code
    assert "COSTS EXCEED EDGE" in out
    assert "not fixable by size or tuning" in out
    assert "NO_GO -- costs exceed edge" in out


def test_surviving_example_is_not_flagged_as_fatal(capsys):
    rc = main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl", "--cost", "fee",
               "--predicted", "model_p", "--won", "won"])
    out = capsys.readouterr().out
    assert rc == 1                                   # 400 trades, cannot tell it from zero
    assert "COSTS EXCEED EDGE" not in out
    assert "CALIBRATION" in out
    assert "NO_GO -- cannot distinguish" in out


def test_split_names_the_worst_bucket_but_calls_it_noise(capsys):
    """binary_trades' worst slice is -38.71 at z=-0.8. It must be shown, and
    it must not be presented as a finding."""
    main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl", "--cost", "fee",
          "--predicted", "model_p", "--won", "won", "--by", "minutes_left"])
    out = capsys.readouterr().out
    assert "SPLIT BY MINUTES_LEFT" in out
    assert "within noise" in out
    assert "in-sample fitting" not in out


def test_calibration_does_not_flag_a_sub_2_sigma_gap(capsys):
    """Every bucket on the shipped examples sits inside two binomial SEs."""
    for name in ("binary_trades.csv", "fee_eaten.csv"):
        main([str(EXAMPLES / name), "--pnl", "pnl", "--cost", "fee",
              "--predicted", "model_p", "--won", "won"])
        out = capsys.readouterr().out
        assert "Overconfident" not in out
        assert "within noise" in out


def test_calibration_flags_a_real_overconfidence(capsys, tmp_path):
    """Model says 90%, wins 70% over 200 trades: z=-9. That is a finding."""
    p = tmp_path / "t.csv"
    rows = ["pnl,p,won"] + [f"1,0.9,{'true' if i % 10 < 7 else 'false'}" for i in range(200)]
    p.write_text("\n".join(rows) + "\n")
    main([str(p), "--pnl", "pnl", "--predicted", "p", "--won", "won"])
    assert ">>> Overconfident" in capsys.readouterr().out


def test_power_refuses_to_size_for_the_observed_mean(capsys):
    """required_n(|mean|) restates the p-value. Without --claimed or --target
    the section must say so instead of printing a number."""
    main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl"])
    out = capsys.readouterr().out
    assert "no target edge" in out
    assert "trades needed" not in out


def test_power_sizes_for_an_explicit_target(capsys):
    main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl", "--target", "0.5"])
    out = capsys.readouterr().out
    assert "to detect 0.5000/trade" in out
    assert "trades needed" in out


def test_unit_cost_without_won_and_price_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(EXAMPLES / "fee_eaten.csv"), "--pnl", "pnl", "--unit-cost", "0.01"])
    assert exc.value.code == 2
    assert "--unit-cost needs --won" in capsys.readouterr().err


def test_unit_cost_with_won_and_price_prints_per_unit_economics(capsys):
    rc = main([str(EXAMPLES / "fee_eaten.csv"), "--pnl", "pnl", "--won", "won",
               "--price", "entry_price", "--unit-cost", "0.01"])
    assert rc == 1
    assert "per unit" in capsys.readouterr().out


def test_a_missing_file_exits_nonzero(capsys):
    assert main(["/nonexistent/nope.csv"]) == 2


def test_a_bad_column_mapping_says_so_rather_than_reporting_zero(capsys, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    rc = main([str(p), "--pnl", "not_a_column"])
    assert rc == 1
    assert "column mapping" in capsys.readouterr().err


# ---- 0.2: subcommands, plans, json, exit codes ------------------------------

def _timed_csv(tmp_path, n=66, per_day=3, mean=2.0, name="t.csv"):
    """n trades, per_day per calendar day, deterministic P&L around `mean`."""
    import random
    rng = random.Random(0)
    lines = ["pnl,fee,ts,size,depth"]
    for i in range(n):
        day = i // per_day
        hour = (i % per_day) * (24 // per_day)          # spread evenly, so no gap looks like an outage
        ts = f"2026-01-{1 + day:02d}T{hour:02d}:00:00Z"
        lines.append(f"{rng.gauss(mean, 1.0):.4f},0.05,{ts},10,{2 if i % 2 else 50}")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_bare_path_still_means_run(tmp_path, capsys):
    p = _timed_csv(tmp_path)
    assert main([str(p), "--pnl", "pnl"]) == main(["run", str(p), "--pnl", "pnl"])


def test_a_clear_edge_is_go_with_exit_zero(tmp_path, capsys):
    rc = main([str(_timed_csv(tmp_path)), "--pnl", "pnl", "--time", "ts", "--cluster", "auto"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "GO -- interval" in out
    assert "cluster-robust" in out
    assert "clusters              22" in out


def test_too_few_clusters_is_no_verdict_with_exit_three(tmp_path, capsys):
    p = _timed_csv(tmp_path, n=60, per_day=10)          # 6 days
    rc = main([str(p), "--pnl", "pnl", "--time", "ts", "--cluster", "auto"])
    assert rc == 3
    assert "NO_VERDICT -- 6 clusters, need 21" in capsys.readouterr().out


def test_json_output_carries_the_verdict_and_exit_code(tmp_path, capsys):
    import json
    p = _timed_csv(tmp_path)
    rc = main([str(p), "--pnl", "pnl", "--time", "ts", "--cluster", "auto", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert d["verdict"]["status"] == "GO" and d["verdict"]["exit_code"] == 0
    assert d["expectancy"]["clusters"] == 22
    assert d["quality"]["findings"] == []
    assert d["edgecheck"] == "0.2.0"


def test_data_section_names_a_recorder_gap(tmp_path, capsys):
    lines = ["pnl,ts"]
    lines += [f"1.0,2026-01-01T10:{i:02d}:00Z" for i in range(30)]
    lines += [f"1.0,2026-01-03T10:{i:02d}:00Z" for i in range(30)]     # two days of silence
    p = tmp_path / "gap.csv"
    p.write_text("\n".join(lines) + "\n")
    main([str(p), "--pnl", "pnl", "--time", "ts"])
    out = capsys.readouterr().out
    assert "0. DATA" in out
    assert "was the recorder running?" in out


def test_dependent_trades_without_cluster_are_capped_to_no_verdict(tmp_path, capsys):
    lines = ["pnl,ts"] + [f"{2.0 + (i % 3) * 0.1},2026-01-01T10:00:00Z" for i in range(100)]
    p = tmp_path / "same_minute.csv"
    p.write_text("\n".join(lines) + "\n")
    rc = main([str(p), "--pnl", "pnl", "--time", "ts"])
    out = capsys.readouterr().out
    assert rc == 3
    assert "NAIVE" in out
    assert "not independent" in out


def test_bankroll_reports_path_and_ruin(tmp_path, capsys):
    rc = main([str(_timed_csv(tmp_path)), "--pnl", "pnl", "--time", "ts", "--bankroll", "20"])
    out = capsys.readouterr().out
    assert "5. PATH" in out and "P(ruin)" in out
    assert rc in (0, 1)


def test_deploy_size_reports_capacity(tmp_path, capsys):
    rc = main([str(_timed_csv(tmp_path)), "--pnl", "pnl", "--size", "size",
               "--depth", "depth", "--deploy-size", "30"])
    out = capsys.readouterr().out
    assert "6. CAPACITY" in out
    assert "depth-limited         33 of 66" in out
    assert rc in (0, 1)


def test_deploy_size_without_depth_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        main([str(_timed_csv(tmp_path)), "--pnl", "pnl", "--deploy-size", "30"])


def test_cluster_auto_without_time_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        main([str(_timed_csv(tmp_path)), "--pnl", "pnl", "--cluster", "auto"])


def test_plan_then_check_refuses_early_and_scores_when_met(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    rc = main(["plan", "--target", "1.0", "--cluster", "auto", "--min-units", "10",
               "--go-threshold", "1.5", "-o", str(plan)])
    assert rc == 0
    assert "sha256" in capsys.readouterr().out

    early = _timed_csv(tmp_path, n=15, per_day=3, name="early.csv")     # 5 days
    rc = main(["check", str(early), "--plan", str(plan), "--pnl", "pnl", "--time", "ts"])
    out = capsys.readouterr().out
    assert rc == 3
    assert "NO_VERDICT -- 5 clusters, need 10" in out
    assert "plan " in out

    later = _timed_csv(tmp_path, n=60, per_day=3, name="later.csv")     # 20 days, mean ~2
    rc = main(["check", str(later), "--plan", str(plan), "--pnl", "pnl", "--time", "ts"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "GO -- interval" in out
    assert "threshold +1.5000" in out


def test_check_applies_the_plan_threshold_not_a_flag(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    main(["plan", "--target", "1.0", "--cluster", "auto", "--min-units", "10",
          "--go-threshold", "5.0", "-o", str(plan)])
    capsys.readouterr()
    p = _timed_csv(tmp_path, n=60, per_day=3)                           # mean ~2 < 5
    rc = main(["check", str(p), "--plan", str(plan), "--pnl", "pnl", "--time", "ts"])
    assert rc == 1
    assert "below threshold +5.0000" in capsys.readouterr().out


def test_check_refuses_an_edited_plan(tmp_path, capsys):
    import json
    plan = tmp_path / "plan.json"
    main(["plan", "--target", "1.0", "-o", str(plan)])
    doc = json.loads(plan.read_text())
    doc["go_threshold"] = -100.0
    plan.write_text(json.dumps(doc))
    rc = main(["check", str(_timed_csv(tmp_path)), "--plan", str(plan), "--pnl", "pnl"])
    assert rc == 2
    assert "edited after it was written" in capsys.readouterr().err


def test_check_json_carries_the_plan_hash(tmp_path, capsys):
    import json
    plan = tmp_path / "plan.json"
    main(["plan", "--target", "1.0", "-o", str(plan)])
    h = json.loads(plan.read_text())["sha256"]
    capsys.readouterr()
    main(["check", str(_timed_csv(tmp_path)), "--plan", str(plan), "--pnl", "pnl", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["plan_hash"] == h
    assert d["verdict"]["rule"]["source"] == f"plan {h[:12]}"
