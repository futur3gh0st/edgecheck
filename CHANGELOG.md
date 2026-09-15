# Changelog

## 0.2.0 — 2026-09-15

Time is the keystone 0.1 lacked. With it, the tool can tell whether the
evidence is independent, whether the recorder was running, and whether the
path to the mean was survivable — and it ends with one verdict instead of
five sections the reader has to reconcile.

### Added
- `--time` and `--cluster` (a column, or `auto` for the UTC day). Trades come
  back in entry order; trades in one cluster are one piece of evidence.
- **Cluster-robust intervals** (CR1 on *G − 1* df) whenever `--cluster` is
  given, printed beside the naive per-trade interval. Simulation tests show
  the naive interval covering the truth ~25% of the time on clustered noise
  while claiming 95%; the cluster interval covers 95–98%.
- `--bootstrap`: block-bootstrap interval, resampling whole clusters.
- **Section 0, Data**: recorder gaps, duplicate keys, trades sharing an entry
  minute with no cluster named, lopsided sides, unreadable fields.
- **Section 5, Path**: max drawdown, longest underwater run; with
  `--bankroll`, drawdown as a share of the stake and P(ruin) by cluster
  resampling.
- **Section 6, Capacity**: `--deploy-size` against `--fill-size` or
  `--depth` — P&L at the size you would run versus what linear scaling
  promised.
- **Section 8, Verdict**: one `GO` / `NO_GO` / `NO_VERDICT` line with every
  reason named and the rule printed. Exit codes carry it: 0 / 1 / 3
  (2 for usage).
- `--json`: the whole analysis as data, same field names as the text.
- `edgecheck plan` writes a frozen protocol (target, clustering, evidence
  floor, go-threshold, drawdown limit) under a sha256 seal;
  `edgecheck check --plan` refuses a verdict until the floor is met, scores
  against the frozen thresholds, and refuses an edited plan.
- Holm-adjusted p-values on calibration and split buckets; a bucket is
  flagged on the adjusted p, not the raw z.
- `--min-units` to raise the evidence floor without a plan.
- Library: `analyse()`, `format_report()`, `to_dict()`, `Options`,
  `Analysis`; `assess()`, `drawdown()`, `ruin_probability()`, `capacity()`,
  `decide()`, `new_plan()` / `write_plan()` / `load_plan()`.

### Changed
- `required_n` re-solves with the *t* critical value, matching the interval.
- Bare `edgecheck trades.csv ...` still works and means `edgecheck run ...`.
- The verdict floor is 30 trades unclustered, **21 clusters** clustered.
- Shipped examples carry a `ts` column (one trade every 90 minutes, ~25 days).

### Exit codes (breaking)
- A successful run no longer exits 0 unconditionally. 0 now means GO.

## 0.1.0 — 2026-09-11

Initial release: costs, expectancy with *t* intervals, calibration, split,
power; flat and paired loaders; refuses a verdict under 30 resolved trades.
