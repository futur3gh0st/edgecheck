"""End-to-end: the shipped examples must produce the verdicts they illustrate."""

from __future__ import annotations

from pathlib import Path

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


def test_split_reports_a_concentrated_loss(capsys):
    main([str(EXAMPLES / "binary_trades.csv"), "--pnl", "pnl", "--cost", "fee",
          "--predicted", "model_p", "--won", "won", "--by", "minutes_left"])
    out = capsys.readouterr().out
    assert "SPLIT BY MINUTES_LEFT" in out


def test_a_missing_file_exits_nonzero(capsys):
    assert main(["/nonexistent/nope.csv"]) == 2


def test_a_bad_column_mapping_says_so_rather_than_reporting_zero(capsys, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    rc = main([str(p), "--pnl", "not_a_column"])
    assert rc == 1
    assert "column mapping" in capsys.readouterr().err
