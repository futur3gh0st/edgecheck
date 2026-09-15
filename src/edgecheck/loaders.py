"""Read trades from whatever shape they already exist in.

The point of this module is that you should not have to reshape your data to
ask whether your strategy works. Point the tool at the CSV or JSONL you already
write, tell it which columns mean what, and it reads them.

Two record shapes are supported:

FLAT    one row per settled trade -- entry and outcome together. Most backtests
        and most exported trade logs look like this.

PAIRED  an entry row and, later, a separate resolve row, joined on a key. Live
        systems that append as things happen look like this, because the
        outcome is not known when the position opens.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ColumnMap:
    """Which fields carry the meaning. Only `pnl` is strictly required.

    predicted  the model's stated probability that this trade wins
    won        whether it did (bool, or 0/1, or "true"/"yes")
    pnl        realised result, net of costs
    cost       what was paid to trade (fees, commission, slippage budget)
    claimed    edge the model claimed at entry, per unit
    size       units traded, used to scale `claimed` into currency
    price      entry price -- for probability-priced instruments this is also
               the breakeven, which is what makes unit economics computable
    side       used with `predicted_is_yes` below
    time       when the trade was entered -- ISO-8601 or epoch seconds/ms.
               Unlocks ordering, drawdown, gap detection and clustering
    cluster    which trades share an outcome: a date, a session, a market.
               Trades in one cluster are not independent evidence. The
               special value "auto" clusters by UTC calendar day of `time`
    fill_size  units actually filled (vs `size` requested), for capacity
    depth      resting size available at entry, for capacity
    """

    pnl: str = "pnl"
    predicted: str | None = None
    won: str | None = None
    cost: str | None = None
    claimed: str | None = None
    size: str | None = None
    price: str | None = None
    side: str | None = None
    time: str | None = None
    cluster: str | None = None
    fill_size: str | None = None
    depth: str | None = None

    # Some systems store P(YES) rather than P(the side actually taken). They are
    # the same number on a yes-side trade and complements on a no-side one, so
    # reading one as the other silently scores half your trades against their
    # own opposite. It is invisible until a no-side trade appears.
    predicted_is_yes: bool = False
    no_side_values: tuple[str, ...] = ("no", "down", "short", "sell")

    # PAIRED only
    key: str | None = None
    kind: str | None = None
    entry_kind: str | None = None
    resolve_kind: str | None = None

    extra: list[str] = field(default_factory=list)

    @property
    def paired(self) -> bool:
        return bool(self.key and self.kind and self.entry_kind and self.resolve_kind)


@dataclass
class Trade:
    """One settled trade, normalised."""

    pnl: float
    predicted: float | None = None
    won: bool | None = None
    cost: float = 0.0
    claimed: float | None = None
    size: float | None = None
    price: float | None = None
    time: float | None = None          # epoch seconds
    cluster: str | None = None
    fill_size: float | None = None
    depth: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def claimed_value(self) -> float | None:
        """Claimed edge scaled into currency, when size is known."""
        if self.claimed is None:
            return None
        return self.claimed * (self.size if self.size is not None else 1.0)


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Epoch values above this are taken to be milliseconds. 1e11 s is the year
# 5138; 1e11 ms is 1973. No real trade log sits in the ambiguous band.
_MS_THRESHOLD = 1e11


def parse_time(v: Any) -> float | None:
    """ISO-8601 (with or without zone; naive is read as UTC) or epoch
    seconds/milliseconds -> epoch seconds. None when it cannot be read."""
    if v is None or v == "":
        return None
    n = _num(v)
    if n is not None:
        return n / 1000.0 if abs(n) > _MS_THRESHOLD else n
    s = str(v).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def day_of(ts: float) -> str:
    """UTC calendar day, the default cluster when `cluster="auto"`."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "win", "won"}:
        return True
    if s in {"0", "false", "f", "no", "n", "loss", "lost"}:
        return False
    n = _num(v)
    return None if n is None else n != 0


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV or JSONL into dicts. Format is taken from the extension, then
    sniffed, because a .txt full of JSON is a normal thing to be handed."""
    text_first = ""
    with path.open() as f:
        for line in f:
            if line.strip():
                text_first = line.strip()
                break
    looks_json = text_first.startswith("{")
    if path.suffix.lower() in {".jsonl", ".ndjson"} or looks_json:
        out = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # tolerate a partial final line on a live file
        return out
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    with path.open() as f:
        return list(csv.DictReader(f))


def _predicted(row: dict[str, Any], m: ColumnMap) -> float | None:
    if not m.predicted:
        return None
    p = _num(row.get(m.predicted))
    if p is None:
        return None
    if m.predicted_is_yes and m.side:
        side = str(row.get(m.side, "")).strip().lower()
        if side in m.no_side_values:
            return 1.0 - p
    return p


def _trade(row: dict[str, Any], m: ColumnMap, pnl_row: dict[str, Any] | None = None) -> Trade | None:
    src = pnl_row if pnl_row is not None else row
    pnl = _num(src.get(m.pnl))
    if pnl is None:
        return None
    return Trade(
        pnl=pnl,
        predicted=_predicted(row, m),
        won=_bool(src.get(m.won)) if m.won else None,
        cost=_num(src.get(m.cost)) or (_num(row.get(m.cost)) or 0.0) if m.cost else 0.0,
        claimed=_num(row.get(m.claimed)) if m.claimed else None,
        size=_num(row.get(m.size)) if m.size else None,
        price=_num(row.get(m.price)) if m.price else None,
        # Entry time comes from the entry row: that is when the decision was
        # made, and clustering, gaps and drawdown are all about decisions.
        time=parse_time(row.get(m.time)) if m.time else None,
        cluster=_cluster(row, m),
        fill_size=_num(src.get(m.fill_size)) if m.fill_size else None,
        depth=_num(row.get(m.depth)) if m.depth else None,
        raw={**row, **(pnl_row or {})},
    )


def _cluster(row: dict[str, Any], m: ColumnMap) -> str | None:
    if not m.cluster:
        return None
    if m.cluster == "auto":
        if not m.time:
            return None
        ts = parse_time(row.get(m.time))
        return day_of(ts) if ts is not None else None
    v = row.get(m.cluster)
    return None if v is None or v == "" else str(v)


def load(path: Path, m: ColumnMap) -> list[Trade]:
    """Read and normalise. Picks flat or paired based on the mapping.

    With a time column, trades come back in entry order -- drawdown and gap
    detection are meaningless otherwise -- and rows whose time cannot be read
    keep file order at the end, so they are not silently dropped.
    """
    rows = read_rows(path)
    if not rows:
        return []
    trades = _load_paired(rows, m) if m.paired else _load_flat(rows, m)
    if m.time:
        timed = [t for t in trades if t.time is not None]
        untimed = [t for t in trades if t.time is None]
        trades = sorted(timed, key=lambda t: t.time or 0.0) + untimed
    return trades


def _load_flat(rows: list[dict[str, Any]], m: ColumnMap) -> list[Trade]:
    out = []
    for r in rows:
        t = _trade(r, m)
        if t is not None:
            out.append(t)
    return out


def _load_paired(rows: list[dict[str, Any]], m: ColumnMap) -> list[Trade]:
    """Join entry rows to resolve rows on `key`.

    The same key can legitimately recur -- the same market traded twice -- so
    entries are consumed in order rather than looked up. Binding every resolve
    to the first entry with that key would silently reuse one entry's claimed
    edge for several outcomes.
    """
    kind, key = m.kind, m.key
    if not (kind and key):
        raise ValueError("paired loading needs both `kind` and `key` columns")
    pending: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if str(r.get(kind)) == m.entry_kind:
            pending.setdefault(str(r.get(key)), []).append(r)
    out = []
    for r in rows:
        if str(r.get(kind)) != m.resolve_kind:
            continue
        queue = pending.get(str(r.get(key)))
        if not queue:
            continue
        entry = queue.pop(0)
        t = _trade(entry, m, pnl_row=r)
        if t is not None:
            out.append(t)
    return out


def iter_numeric(trades: list[Trade], column: str) -> Iterator[tuple[float, Trade]]:
    """Yield (value, trade) for a column, skipping rows where it is absent."""
    for t in trades:
        v = _num(t.raw.get(column))
        if v is not None:
            yield v, t
