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
    assert rc == 0
    assert "COSTS EXCEED EDGE" in out
    assert "not fixable by size or tuning" in out


def test_surviving_example_is_not_flagged_as_fatal(capsys):
    rc = main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl", "--cost", "fee",
               "--predicted", "model_p", "--won", "won"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "COSTS EXCEED EDGE" not in out
    assert "CALIBRATION" in out


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
    assert rc == 0
    assert "per unit" in capsys.readouterr().out


def test_a_missing_file_exits_nonzero(capsys):
    assert main(["/nonexistent/nope.csv"]) == 2


def test_a_bad_column_mapping_says_so_rather_than_reporting_zero(capsys, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    rc = main([str(p), "--pnl", "not_a_column"])
    assert rc == 1
    assert "column mapping" in capsys.readouterr().err
