"""Command line entry point.

    edgecheck run   trades.csv [columns] [analysis]     the report
    edgecheck       trades.csv ...                      same (0.1 invocations still work)
    edgecheck plan  --target E [...] -o plan.json       freeze a protocol
    edgecheck check trades.csv --plan plan.json [...]   score against it

Exit codes carry the verdict so a script cannot wave it through:
    0  GO        1  NO_GO        3  NO_VERDICT        2  usage / cannot read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fees import unit_economics
from .loaders import ColumnMap, load
from .plan import Plan, PlanError, load_plan, new_plan, write_plan
from .report import Options, analyse, format_report, to_dict

EPILOG = """\
examples:
  # a flat CSV of settled trades, clustered by calendar day
  edgecheck trades.csv --pnl pnl --cost fee --predicted model_p --won won \\
      --time ts --cluster auto

  # claimed edge per unit, scaled by size, to get the realisation ratio
  edgecheck trades.csv --pnl pnl --cost fee --claimed edge --size shares

  # would the edge survive $15 a side, given the depth the log recorded?
  edgecheck trades.csv --pnl pnl --size size --depth depth --deploy-size 15

  # would you have survived the path on a $1,000 stake?
  edgecheck trades.csv --pnl pnl --time ts --bankroll 1000

  # freeze the protocol first, then score against it
  edgecheck plan --target 0.05 --cluster auto --min-units 21 --go-threshold 10 -o plan.json
  edgecheck check live.jsonl --plan plan.json --pnl pnl --time ts --json

  # a live log that appends an entry row and a resolve row separately
  edgecheck live.jsonl --pnl pnl_after_fee --kind kind \\
      --entry-kind open --resolve-kind settle --key ticker

  # model stores P(YES) rather than P(side taken)
  edgecheck trades.csv --pnl pnl --predicted fair --side side --predicted-is-yes
"""

RUN_COMMANDS = {"run", "check"}


def _add_columns(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("columns")
    g.add_argument("--pnl", default="pnl", help="realised result, net of costs (default: pnl)")
    g.add_argument("--cost", help="cost paid to trade -- fees, commission, slippage")
    g.add_argument("--predicted", help="model's probability this trade wins")
    g.add_argument("--won", help="whether it won")
    g.add_argument("--claimed", help="edge the model claimed at entry, per unit")
    g.add_argument("--size", help="units requested")
    g.add_argument("--fill-size", help="units actually filled (for --deploy-size)")
    g.add_argument("--depth", help="resting size at entry (for --deploy-size)")
    g.add_argument("--price", help="entry price (breakeven, for probability-priced instruments)")
    g.add_argument("--side", help="side taken, used with --predicted-is-yes")
    g.add_argument("--time", help="entry time: ISO-8601 or epoch seconds/ms")
    g.add_argument("--cluster", metavar="COLUMN|auto",
                   help="what trades share an outcome: a column, or 'auto' for the UTC day "
                        "of --time. Trades in one cluster are one piece of evidence")
    g.add_argument("--predicted-is-yes", action="store_true",
                   help="--predicted holds P(YES), not P(side taken). Without this a "
                        "no-side trade is scored against its own complement.")

    pr = p.add_argument_group("paired logs (entry row + later resolve row)")
    pr.add_argument("--kind", help="column naming the row type")
    pr.add_argument("--entry-kind", help="value of --kind marking an entry")
    pr.add_argument("--resolve-kind", help="value of --kind marking a settlement")
    pr.add_argument("--key", help="column joining an entry to its resolve")


def _add_analysis(p: argparse.ArgumentParser, with_plan_overrides: bool) -> None:
    o = p.add_argument_group("analysis")
    o.add_argument("--by", metavar="COLUMN", help="also split results by a numeric column")
    o.add_argument("--buckets", type=int, default=5, help="buckets per split (default 5)")
    o.add_argument("--bankroll", type=float, metavar="AMOUNT",
                   help="stake, to report drawdown as a share of it and P(ruin)")
    o.add_argument("--deploy-size", type=float, metavar="UNITS",
                   help="size you would run; needs --size and --fill-size or --depth")
    o.add_argument("--bootstrap", action="store_true",
                   help="also print a block-bootstrap interval (2,000 resamples)")
    o.add_argument("--unit-cost", type=float,
                   help="cost per unit, to compare against edge per unit "
                        "(requires --won and --price)")
    o.add_argument("--unit", default="contract", help="name of the unit (default: contract)")
    o.add_argument("--json", action="store_true", help="emit the analysis as JSON instead of text")
    if with_plan_overrides:
        o.add_argument("--target", type=float, metavar="EDGE",
                       help="per-trade edge to size the power analysis for, when no "
                            "--claimed column is logged")
        o.add_argument("--min-units", type=int, metavar="N",
                       help="evidence floor for a verdict: trades, or clusters with --cluster "
                            "(default 30 / 21)")


def _add_plan_fields(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", type=float, required=True, metavar="EDGE",
                   help="per-trade edge the run is sized to detect")
    p.add_argument("--cluster", metavar="COLUMN|auto",
                   help="what trades share an outcome (see run --cluster)")
    p.add_argument("--min-units", type=int, metavar="N",
                   help="evidence floor: trades, or clusters when clustered (default 30 / 21)")
    p.add_argument("--go-threshold", type=float, default=0.0, metavar="EDGE",
                   help="mean per trade required for GO (default 0)")
    p.add_argument("--max-drawdown-frac", type=float, default=0.5, metavar="F",
                   help="max drawdown allowed as a share of --bankroll (default 0.5)")
    p.add_argument("--bankroll", type=float, help="stake the drawdown limit is judged against")
    p.add_argument("--note", default="", help="free text recorded in the plan")
    p.add_argument("-o", "--out", type=Path, required=True, help="where to write the plan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edgecheck",
        description="Does the claimed edge survive contact with reality?",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="analyse a file of settled trades",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_argument("path", type=Path, help="CSV, JSON or JSONL of trades")
    _add_columns(run)
    _add_analysis(run, with_plan_overrides=True)

    plan = sub.add_parser("plan", help="freeze a protocol before the data arrives")
    _add_plan_fields(plan)

    check = sub.add_parser("check", help="score a file against a frozen plan")
    check.add_argument("path", type=Path, help="CSV, JSON or JSONL of trades")
    check.add_argument("--plan", type=Path, required=True, help="plan written by `edgecheck plan`")
    _add_columns(check)
    _add_analysis(check, with_plan_overrides=False)
    return p


def _normalise(argv: list[str] | None) -> list[str]:
    """`edgecheck trades.csv ...` means `edgecheck run trades.csv ...`."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in {"run", "plan", "check", "-h", "--help"}:
        return ["run", *args]
    return args


def _column_map(args: argparse.Namespace) -> ColumnMap:
    return ColumnMap(
        pnl=args.pnl, cost=args.cost, predicted=args.predicted, won=args.won,
        claimed=args.claimed, size=args.size, price=args.price, side=args.side,
        time=args.time, cluster=args.cluster, fill_size=args.fill_size, depth=args.depth,
        predicted_is_yes=args.predicted_is_yes,
        kind=args.kind, entry_kind=args.entry_kind,
        resolve_kind=args.resolve_kind, key=args.key,
    )


def _plan_command(args: argparse.Namespace) -> int:
    plan = new_plan(
        target=args.target, cluster=args.cluster, min_units=args.min_units,
        go_threshold=args.go_threshold, max_drawdown_frac=args.max_drawdown_frac,
        bankroll=args.bankroll, note=args.note,
    )
    h = write_plan(plan, args.out)
    print(f"  plan written to {args.out}")
    print(f"  sha256 {h}")
    print(f"  target {plan.target:+.4f}/trade, min {plan.effective_min_units} "
          f"{'clusters' if plan.cluster else 'trades'}, go at mean >= {plan.go_threshold:+.4f}")
    print("  commit this file before the data arrives.")
    return 0


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.unit_cost is not None and not (args.won and args.price):
        parser.error("--unit-cost needs --won (for the realised win rate) "
                     "and --price (for the breakeven)")
    if args.deploy_size is not None and not (args.size and (args.fill_size or args.depth)):
        parser.error("--deploy-size needs --size and one of --fill-size / --depth")
    if args.cluster == "auto" and not args.time:
        parser.error("--cluster auto needs --time")
    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    plan: Plan | None = None
    if args.command == "check":
        try:
            plan = load_plan(args.plan)
        except PlanError as exc:
            print(f"  {exc}", file=sys.stderr)
            return 2
        # The plan's clustering and bankroll are part of the protocol and win
        # over anything typed on the command line.
        if plan.cluster:
            args.cluster = plan.cluster
        if plan.bankroll is not None:
            args.bankroll = plan.bankroll
        if args.cluster == "auto" and not args.time:
            parser.error("the plan clusters by 'auto', which needs --time")

    m = _column_map(args)
    trades = load(args.path, m)
    if not trades:
        print("  No resolved trades found. Check the column mapping "
              "(--pnl names the result column).", file=sys.stderr)
        return 1

    ue = None
    if args.unit_cost is not None:
        scored = [t for t in trades if t.won is not None and t.price is not None]
        if not scored:
            print(f"  --unit-cost: no rows carry both '{args.won}' and '{args.price}'.",
                  file=sys.stderr)
            return 1
        prices = [t.price for t in scored if t.price is not None]
        ue = unit_economics(sum(1 for t in scored if t.won) / len(scored),
                            sum(prices) / len(prices), args.unit_cost, args.unit)

    opt = Options(
        split_by=args.by, buckets=args.buckets, unit_econ=ue,
        bankroll=args.bankroll, deploy_size=args.deploy_size, bootstrap=args.bootstrap,
        target=plan.target if plan else args.target,
        min_units=plan.effective_min_units if plan else args.min_units,
        rule=plan.rule() if plan else None,
    )
    a = analyse(trades, m, opt)
    if a is None:
        print("  No resolved trades found. Nothing to measure.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(to_dict(a, plan.sha256 if plan else None), indent=2))
    else:
        if plan:
            print(f"  plan {plan.short_hash}  target {plan.target:+.4f}/trade  "
                  f"created {plan.created}")
        print(format_report(a, deploy_size=args.deploy_size))
    return a.verdict.status.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalise(argv))
    if args.command == "plan":
        return _plan_command(args)
    if args.command in RUN_COMMANDS:
        return _run_command(args, parser)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
