"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fees import unit_economics
from .loaders import ColumnMap, load
from .report import render

EPILOG = """\
examples:
  # a flat CSV of settled trades
  edgecheck trades.csv --pnl pnl --cost fee --predicted model_p --won won

  # claimed edge per unit, scaled by size, to get the realisation ratio
  edgecheck trades.csv --pnl pnl --cost fee --claimed edge --size shares

  # find out whether the loss lives in one slice
  edgecheck trades.csv --pnl pnl --by minutes_left

  # a live log that appends an entry row and a resolve row separately
  edgecheck live.jsonl --pnl pnl_after_fee --kind kind \\
      --entry-kind open --resolve-kind settle --key ticker

  # model stores P(YES) rather than P(side taken)
  edgecheck trades.csv --pnl pnl --predicted fair --side side --predicted-is-yes
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edgecheck",
        description="Does the claimed edge survive contact with reality?",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path, help="CSV, JSON or JSONL of trades")

    g = p.add_argument_group("columns")
    g.add_argument("--pnl", default="pnl", help="realised result, net of costs (default: pnl)")
    g.add_argument("--cost", help="cost paid to trade -- fees, commission, slippage")
    g.add_argument("--predicted", help="model's probability this trade wins")
    g.add_argument("--won", help="whether it won")
    g.add_argument("--claimed", help="edge the model claimed at entry, per unit")
    g.add_argument("--size", help="units traded, to scale --claimed into currency")
    g.add_argument("--price", help="entry price (breakeven, for probability-priced instruments)")
    g.add_argument("--side", help="side taken, used with --predicted-is-yes")
    g.add_argument("--predicted-is-yes", action="store_true",
                   help="--predicted holds P(YES), not P(side taken). Without this a "
                        "no-side trade is scored against its own complement.")

    pr = p.add_argument_group("paired logs (entry row + later resolve row)")
    pr.add_argument("--kind", help="column naming the row type")
    pr.add_argument("--entry-kind", help="value of --kind marking an entry")
    pr.add_argument("--resolve-kind", help="value of --kind marking a settlement")
    pr.add_argument("--key", help="column joining an entry to its resolve")

    o = p.add_argument_group("analysis")
    o.add_argument("--by", metavar="COLUMN", help="also split results by a numeric column")
    o.add_argument("--buckets", type=int, default=5, help="buckets per split (default 5)")
    o.add_argument("--unit-cost", type=float,
                   help="cost per unit, to compare against edge per unit")
    o.add_argument("--unit", default="contract", help="name of the unit (default: contract)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    m = ColumnMap(
        pnl=args.pnl, cost=args.cost, predicted=args.predicted, won=args.won,
        claimed=args.claimed, size=args.size, price=args.price, side=args.side,
        predicted_is_yes=args.predicted_is_yes,
        kind=args.kind, entry_kind=args.entry_kind,
        resolve_kind=args.resolve_kind, key=args.key,
    )
    trades = load(args.path, m)
    if not trades:
        print("  No resolved trades found. Check the column mapping "
              "(--pnl names the result column).", file=sys.stderr)
        return 1

    ue = None
    if args.unit_cost is not None:
        wins = [t for t in trades if t.won]
        prices = [t.price for t in trades if t.price is not None]
        if prices:
            ue = unit_economics(len(wins) / len(trades),
                                sum(prices) / len(prices),
                                args.unit_cost, args.unit)

    print(render(trades, split_by=args.by, unit_econ=ue, buckets=args.buckets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
