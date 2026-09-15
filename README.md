# edgecheck

**Does your strategy's claimed edge survive contact with reality?**

Point it at a file of settled trades. It tells you, in order, the things that
actually kill strategies — and ends with one verdict, the rule that produced
it, and an exit code that carries it, so a script cannot wave it through.

```
pip install edgecheck
edgecheck trades.csv --pnl pnl --cost fee --time ts --cluster auto
```

```
  8. VERDICT
  --------------------------------------------------------------
  NO_GO -- cannot distinguish the edge from zero
  rule                  default; drawdown <= 50% of bankroll
```

Exit code 1. Or, on a strategy that pays more to trade than the trade is worth:

```
  1. COSTS -- does the edge survive the toll?
  --------------------------------------------------------------
  gross, before costs      +25.69
  costs paid               -51.41
  net                      -25.72
  costs / gross edge        200%
  verdict    COSTS EXCEED EDGE -- pays 200% of gross edge to trade;
             not fixable by size or tuning
```

That took one command and no backtest.

---

## Why this exists

Most strategy analysis reports a P&L number and a win rate. Neither answers
the question you actually have, and both are easy to fool yourself with:

- **A P&L near zero is usually not "roughly break-even."** It is "we cannot
  tell yet." Those are completely different, and only a confidence interval
  separates them.
- **A win rate is meaningless without its breakeven.** A high win rate sounds
  excellent, and loses money whenever the price you paid already implied it.
- **A few wins on high-probability bets look exactly like skill.** Small
  samples produce narrow intervals and false confidence.
- **Trades that share an outcome are not independent evidence.** Two hundred
  fills on the same day's weather market are one bet, not two hundred. A
  per-trade interval on that data claims 95% and delivers about 25%.
- **A log with holes in it produces a P&L about the holes.** A recorder that
  slept through most of a window still writes a file that parses.
- **An edge measured at $2 a side does not scale to $15 a side** if the book
  only ever held $2.

edgecheck was built by running a trading bot for eight months, watching a
backtest claim +57%, and finding out — one section at a time — that the
number was an artefact. Every check here is one the bot needed and did not
have. The bot is [public](https://github.com/futur3gh0st/crypto-bot); so is
the report that killed it.

---

## What it reports

Nine sections, ordered by what reaches a decision fastest.

### 0. Data — is this fit to be measured?

Runs before any statistic. With a `--time` column it finds recorder gaps
(longer than ten times the median spacing, reported as a share of the span),
trades sharing an entry minute with no `--cluster` named, unreadable
timestamps. With a `--key` it finds duplicates. With a `--side` it spots the
P(YES)-read-as-P(side) trap. Findings are printed; nothing is silently fixed.

### 1. Costs — the fastest way to a decision

Splits realised results into **gross / costs / net** and compares the edge
you capture against the toll you pay for it. When costs exceed gross edge, it
says so plainly and explains why nothing fixes it: costs scale with size, and
calibration makes a model *honest* about a small edge rather than enlarging it.

### 2. Expectancy — result or noise?

Mean per trade with an interval sized for the evidence you actually have.
With `--cluster`, that is the **cluster-robust** interval on *G − 1* degrees
of freedom, where *G* is the number of clusters — days, sessions, markets —
and the verdict floor counts clusters, not trades. The naive per-trade
interval is printed beside it so you can see what you would have believed.

```
  resolved trades       400
  clusters              26   (the unit of evidence)
  95% CI (cluster-robust, df=25)  [-0.3060, +0.5531]
  naive per-trade CI    [-0.4146, +0.6617]   (what 0.1 would have printed)
  95% bootstrap         [-0.2903, +0.5109]   (blocked by cluster)
```

If you log the edge your model claimed at entry, it also reports the
**realisation ratio** — claimed versus delivered. A model that claims a large
edge and delivers a sliver of it has a different problem from one that simply
loses, and the ratio is what tells them apart.

### 3. Calibration — *where* the model is wrong

Buckets trades by the model's own stated probability and compares it to what
happened. A bucket is only called **overconfident** when its gap survives a
Holm correction for the number of buckets looked at; below that the worst gap
is named and labelled as noise. Looking at five buckets and reporting the
worst is five tests, not one, and the arithmetic now says so.

### 4. Split — is the loss concentrated?

`--by <column>` splits results by any numeric field you log. Same Holm gate.
And when a bucket does clear it, the report warns you: cutting the worst
bucket *after* seeing which bucket was worst is in-sample fitting, and it is
how backtests get retracted.

### 5. Path — would you have survived the way there?

Max drawdown, where it happened, the longest stretch under water. With
`--bankroll`, the drawdown as a share of the stake and **P(ruin)** — the
observed P&L replayed by resampling whole clusters, counting how often the
stake is exhausted. Not a model of the strategy; the sample, replayed.

### 6. Capacity — does the edge survive the size you would run?

Given `--size` and either `--fill-size` (what filled) or `--depth` (what was
resting), `--deploy-size` rescales every trade to what the recorded book could
have provided and reports the P&L at that size against what linear scaling
would have promised. Ten trades at $2 into $2 of depth stay +10 when
"deployed" at $15; linear scaling would have claimed +75.

### 7. Power — how much data would settle it?

Trades needed to detect an edge of a given size at 95% confidence and 80%
power. The target must come from outside the sample — the edge the model
claimed (`--claimed`), or one you name (`--target`). It never sizes for the
observed mean: that only restates the p-value above, and would tell a
pure-noise strategy exactly how many more trades it needs to look significant.

### 8. Verdict

One line, and the rule that produced it.

```
NO_VERDICT  not enough independent evidence, or the evidence is not
            independent and nobody said how it clusters
NO_GO       structurally unprofitable, or the interval sits below zero,
            or the mean misses the threshold, or the drawdown would have
            exhausted the stated bankroll
GO          interval above zero, mean at threshold, drawdown survivable
NO_GO       enough evidence, and it cannot tell the edge from zero
```

Exit codes: **0** GO · **1** NO_GO · **3** NO_VERDICT · **2** usage. `--json`
emits the whole analysis as data with the same field names.

---

## Pre-registration: freeze the protocol before the data arrives

The cleanest way to fool yourself is to pick the threshold after seeing the
number. `plan` writes the threshold down first:

```bash
edgecheck plan --target 0.05 --cluster auto --min-units 21 \
    --go-threshold 10 --bankroll 1000 -o plan.json
git add plan.json && git commit -m "pre-register the forward test"
```

```
  plan written to plan.json
  sha256 3f9c2a7e1b0d...
  target +0.0500/trade, min 21 clusters, go at mean >= +10.0000
  commit this file before the data arrives.
```

Then, as the log grows:

```bash
edgecheck check live.jsonl --plan plan.json --pnl pnl --time ts
```

`check` refuses to render a verdict until the plan's evidence floor is met,
scores against the frozen thresholds — the plan's clustering, threshold and
bankroll override anything typed on the command line — and prints the plan's
hash on the report. A plan file whose hash no longer matches its contents is
refused. The hash is not security; anyone can edit and rehash. It is a
tamper-evident seal for the honest: a report that names hash `3f9c2a7e` can
be checked against a plan committed on a date, and the diff is the story.

---

## Where it's most helpful

| Situation | What it gives you |
|---|---|
| **Evaluating a new strategy before funding it** | A cost decomposition that often ends the discussion in one command |
| **A backtest that looks too good** | Realisation ratio and calibration expose modelled-fill optimism |
| **A paper run sitting near zero** | Separates "no edge" from "not enough data yet" — with a number for how much more |
| **A live log with hundreds of fills a day** | Cluster-robust intervals that count days, not fills |
| **A recorder you are not sure was up** | The data section names the gaps before you trust a number |
| **Deciding what size to run** | Capacity, from the depth your own log recorded |
| **A forward test you want to be believed** | A pre-registered plan, and a verdict scored against it |
| **CI for a trading repo** | `--json` and exit codes: fail the build on NO_GO |

**Where it won't help:** it measures strategies that have *already produced
settled outcomes*. It does not generate signals, optimise parameters, or
predict anything. It is a thermometer, not a furnace.

---

## Input formats

Two shapes, no reshaping required.

**Flat** — one row per settled trade. Most backtests and exports look like this.

```bash
edgecheck trades.csv --pnl pnl --cost fee --predicted model_p --won won --time ts
```

**Paired** — an entry row and a later resolve row, joined on a key. Live
systems that append as things happen look like this.

```bash
edgecheck live.jsonl --pnl pnl_after_fee --time ts \
    --kind kind --entry-kind open --resolve-kind settle --key ticker
```

CSV, JSON and JSONL are all read. A truncated final line — normal when
reading a log that is still being written — is tolerated rather than fatal.
Timestamps are ISO-8601 (naive is read as UTC) or epoch seconds/milliseconds.

### Columns

Only `--pnl` is required. Everything else unlocks another section.

| Flag | Meaning | Unlocks |
|---|---|---|
| `--pnl` | realised result, net of costs | everything |
| `--cost` | fees, commission, slippage | cost decomposition |
| `--time` | entry time | ordering, gaps, drawdown, `--cluster auto` |
| `--cluster` | a column, or `auto` for the UTC day of `--time` | cluster-robust inference |
| `--predicted` | model's probability this trade wins | calibration |
| `--won` | whether it did | calibration |
| `--claimed` | edge claimed at entry, per unit | realisation ratio, power target |
| `--size` | units requested | scales `--claimed`; capacity |
| `--fill-size` / `--depth` | what filled / what was resting | capacity, with `--deploy-size` |
| `--price` | entry price (breakeven, for probability-priced instruments) | unit economics, with `--unit-cost` |
| `--key` | trade identifier | duplicate detection; paired join |
| `--by` | any numeric column | dimensional split |

And the ones that are not columns:

| Flag | Meaning |
|---|---|
| `--bankroll AMOUNT` | stake, for drawdown share and P(ruin) |
| `--deploy-size UNITS` | size you would run, for capacity |
| `--target EDGE` | per-trade edge to size the power analysis for |
| `--min-units N` | evidence floor for a verdict (default 30 trades, or 21 clusters) |
| `--bootstrap` | also print a block-bootstrap interval |
| `--buckets N` | buckets per calibration table and split (default 5) |
| `--unit-cost C` / `--unit NAME` | cost per unit, against realised edge per unit |
| `--json` | the analysis as data |

### One trap worth knowing

Some systems store **P(YES)** rather than P(the side actually taken). They are
the same number on a yes-side trade and complements on a no-side one — so
reading one as the other silently scores half your trades against their own
opposite, and stays invisible until a no-side trade appears. The data section
flags a lopsided side split for this reason. If your data is that shape:

```bash
edgecheck trades.csv --pnl pnl --predicted fair --side side --predicted-is-yes
```

---

## As a library

```python
from pathlib import Path
from edgecheck import ColumnMap, Options, analyse, format_report, load, to_dict

m = ColumnMap(pnl="pnl", cost="fee", predicted="model_p", won="won",
              time="ts", cluster="auto")
trades = load(Path("trades.csv"), m)

a = analyse(trades, m, Options(bankroll=1000, bootstrap=True))
print(a.verdict.status, a.verdict.reasons)        # Status.NO_GO ['cannot distinguish ...']
print(a.expectancy.lo, a.expectancy.hi, a.expectancy.clusters)
print(format_report(a))                          # the text report
to_dict(a)                                       # the JSON shape
```

Or, at a lower level:

```python
from edgecheck import decompose, expectancy, drawdown, capacity

costs = decompose([t.pnl for t in trades], [t.cost for t in trades])
if costs.structurally_unprofitable:
    print(costs.verdict)                         # stop here

e = expectancy([t.pnl for t in trades], [t.cluster for t in trades])
dd = drawdown([t.pnl for t in trades])
```

---

## Try it

Two synthetic datasets ship with the repo:

```bash
# costs survivable (52% of gross), 26 days of evidence, and still not
# enough to tell the edge from zero -- the honest answer is NO_GO
edgecheck examples/binary_trades.csv --pnl pnl --cost fee \
    --predicted model_p --won won --time ts --cluster auto --bankroll 200

# an edge smaller than the fee -- structurally unprofitable
edgecheck examples/fee_eaten.csv --pnl pnl --cost fee \
    --predicted model_p --won won --time ts --cluster auto
```

---

## Install

```bash
pip install edgecheck
```

Python 3.10+. **No dependencies** — standard library only.

From source:

```bash
git clone https://github.com/futur3gh0st/edgecheck
cd edgecheck && pip install -e '.[dev]' && ruff check . && mypy && pytest
```

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT
