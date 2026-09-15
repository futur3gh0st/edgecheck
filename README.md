# edgecheck

**Does your strategy's claimed edge survive contact with reality?**

Point it at a file of settled trades. It tells you, in order, the things that
actually kill strategies — costs first, because a strategy whose toll exceeds
its edge is finished no matter what the statistics say.

```
pip install edgecheck
edgecheck trades.csv --pnl pnl --cost fee --predicted model_p --won won
```

```
  1. COSTS -- does the edge survive the toll?
  --------------------------------------------------------------
  gross, before costs      +25.69
  costs paid               -51.41
  net                      -25.72
  costs / gross edge        200%
  verdict    COSTS EXCEED EDGE -- pays 200% of gross edge to trade;
             not fixable by size or tuning

  >>> More capital scales costs too. Better calibration makes the
      model honest about a small edge, it does not enlarge it.
      Stop here unless the cost side can change.
```

That took one command and no backtest.

---

## Why this exists

Most strategy analysis reports a P&L number and a win rate. Neither answers the
question you actually have, and both are easy to fool yourself with:

- **A P&L near zero is usually not "roughly break-even."** It is "we cannot
  tell yet." Those are completely different, and only a confidence interval
  separates them.
- **A win rate is meaningless without its breakeven.** A high win rate sounds
  excellent, and loses money whenever the price you paid already implied it.
- **A few wins on high-probability bets look exactly like skill.** Small
  samples produce narrow intervals and false confidence.

edgecheck refuses to render a verdict below 30 resolved trades, uses Student's
t rather than a normal approximation (at n=3 the correct multiplier is 4.30,
not 1.96), and — given an edge to test for — reports how much more data would
settle the question.

---

## What it reports

### 1. Costs — the fastest way to a decision

Splits realised results into **gross / costs / net** and compares the edge you
capture against the toll you pay for it. When costs exceed gross edge, it says
so plainly and explains why nothing fixes it: costs scale with size, and
calibration makes a model *honest* about a small edge rather than enlarging it.

Run this first. It ends many projects in one line.

### 2. Expectancy — result or noise?

Mean per trade with a t-interval sized for the sample you actually have, plus a
verdict that requires *both* an interval excluding zero *and* enough trades to
believe it.

If you log the edge your model claimed at entry, it also reports the
**realisation ratio** — claimed versus delivered. This is the single most
revealing number in the tool and almost nobody computes it. A model that
claims a large edge and delivers a sliver of it has a different problem from
one that simply loses, and the ratio is what tells them apart.

### 3. Calibration — *where* the model is wrong

Buckets trades by the model's own stated probability and compares it to what
actually happened. A model with real edge is calibrated: things it calls 70%
happen about 70% of the time.

```
    model says   actually won       n       gap      z
        83.2%         81.2%      80    -2.0%   -0.5
        86.4%         83.8%      80    -2.7%   -0.7
        89.5%         91.2%      80    +1.7%   +0.5
        92.5%         91.2%      80    -1.2%   -0.4
        95.6%         97.5%      80    +1.9%   +0.8
  worst bucket says 86.4%, wins 83.8% (n=80, z=-0.69): within noise, |z| < 2.
```

(Output from `examples/binary_trades.csv`, which ships with the repo. A
well-calibrated model looks like this: gaps scattered around zero.) The `z`
column is the gap in binomial standard errors under the model's own
probability. A bucket is only called **overconfident** when |z| ≥ 2; below
that the worst gap is still named, and labelled as noise. At n=80 a 2.7-point
gap is inside one standard error, and a tool that flagged it would be
committing the exact overclaim it exists to prevent. Every bucket sitting
*significantly* below the line is the signature of systematic overconfidence —
a defect you can locate, rather than a number you can only regret.

### 4. Split — is the loss concentrated?

`--by <column>` splits results by any numeric field you log: time to expiry,
position size, hour of day, volatility regime. If one slice carries the whole
loss, that is worth knowing.

The same |z| ≥ 2 gate applies, using the bucket's summed P&L against the whole
sample's per-trade standard deviation. A merely negative bucket is named and
labelled as noise; only a bucket that clears the gate is called a concentrated
loss — and then it warns you not to act on it. Cutting the worst bucket *after*
seeing which bucket was worst is in-sample fitting, and it is how backtests get
retracted.

### 5. Power — how much data would settle it?

Tells you how many resolved trades are needed to detect an edge of a given
size at 95% confidence and 80% power — so you can size a paper run before it
starts instead of arguing about it afterwards.

The target must come from outside the sample: the edge the model claimed
(`--claimed`), or one you name (`--target 0.05`). It never sizes for the
observed mean. That number is `(z_α + z_β)² / t² · n` — a restatement of the
p-value already printed in section 2 — and it would tell a pure-noise strategy
exactly how many more trades it needs to look significant.

---

## Where it's most helpful

| Situation | What it gives you |
|---|---|
| **Evaluating a new strategy before funding it** | A cost decomposition that often ends the discussion in one command |
| **A backtest that looks too good** | Realisation ratio and calibration expose modelled-fill optimism |
| **A paper run sitting near zero** | Separates "no edge" from "not enough data yet" — with a number for how much more |
| **A model you suspect is overconfident** | Calibration buckets localise the error instead of just confirming it |
| **Deciding how long to run a test** | Power analysis sizes the run in advance |
| **Comparing two variants** | Same measurement applied identically to both |

**Where it won't help:** it measures strategies that have *already produced
settled outcomes*. It does not generate signals, optimise parameters, or
predict anything. It is a thermometer, not a furnace.

---

## Input formats

Two shapes, no reshaping required.

**Flat** — one row per settled trade. Most backtests and exports look like this.

```bash
edgecheck trades.csv --pnl pnl --cost fee --predicted model_p --won won
```

**Paired** — an entry row and a later resolve row, joined on a key. Live systems
that append as things happen look like this, because the outcome is unknown when
the position opens.

```bash
edgecheck live.jsonl --pnl pnl_after_fee \
    --kind kind --entry-kind open --resolve-kind settle --key ticker
```

CSV, JSON and JSONL are all read. A truncated final line — normal when reading a
log that is still being written — is tolerated rather than fatal.

### Columns

Only `--pnl` is required. Everything else unlocks another section.

| Flag | Meaning | Unlocks |
|---|---|---|
| `--pnl` | realised result, net of costs | everything |
| `--cost` | fees, commission, slippage | cost decomposition |
| `--predicted` | model's probability this trade wins | calibration |
| `--won` | whether it did | calibration |
| `--claimed` | edge claimed at entry, per unit | realisation ratio |
| `--size` | units traded | scales `--claimed` into currency |
| `--price` | entry price (breakeven, for probability-priced instruments) | unit economics, with `--unit-cost` |
| `--by` | any numeric column | dimensional split |

And four that are not columns:

| Flag | Meaning |
|---|---|
| `--target EDGE` | per-trade edge to size the power analysis for, when there is no `--claimed` column |
| `--unit-cost C` | cost per unit, compared against realised edge per unit; needs `--won` and `--price` |
| `--unit NAME` | what to call the unit in that comparison (default `contract`) |
| `--buckets N` | buckets per calibration table and split (default 5) |

### One trap worth knowing

Some systems store **P(YES)** rather than P(the side actually taken). They are
the same number on a yes-side trade and complements on a no-side one — so
reading one as the other silently scores half your trades against their own
opposite, and stays invisible until a no-side trade appears.

If your data is that shape, say so:

```bash
edgecheck trades.csv --pnl pnl --predicted fair --side side --predicted-is-yes
```

---

## As a library

```python
from pathlib import Path
from edgecheck import ColumnMap, load, expectancy, decompose, render

trades = load(Path("trades.csv"), ColumnMap(pnl="pnl", cost="fee",
                                            predicted="model_p", won="won"))

costs = decompose([t.pnl for t in trades], [t.cost for t in trades])
if costs.structurally_unprofitable:
    print(costs.verdict)          # stop here

e = expectancy([t.pnl for t in trades])
print(e.mean, e.lo, e.hi, e.verdict)

print(render(trades, split_by="minutes_left"))
```

---

## Try it

Two synthetic datasets ship with the repo:

```bash
# costs survivable (52% of gross), but 400 trades cannot yet tell the
# edge from zero -- the honest answer is "not enough data"
edgecheck examples/binary_trades.csv --pnl pnl --cost fee \
    --predicted model_p --won won --by minutes_left

# an edge smaller than the fee -- structurally unprofitable
edgecheck examples/fee_eaten.csv --pnl pnl --cost fee \
    --predicted model_p --won won
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

## License

MIT
