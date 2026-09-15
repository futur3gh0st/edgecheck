"""edgecheck -- does your strategy's claimed edge survive contact with reality?

Point it at a file of settled trades and it answers, in order:

  0. Is the data fit to be measured?   (gaps, duplicates, dependence)
  1. Do costs exceed the gross edge?   (if yes, nothing else matters)
  2. Is the result distinguishable from noise?   (cluster-robust)
  3. Where is the model's probability wrong?
  4. Is the loss concentrated in one slice?
  5. Would you have survived the path?
  6. Does the edge survive the size you would run?
  7. How much more data would settle it?
  8. One verdict, and the rule that produced it.
"""

from .capacity import Capacity, capacity
from .fees import CostBreakdown, UnitEconomics, decompose, unit_economics
from .loaders import ColumnMap, Trade, load, parse_time, read_rows
from .plan import Plan, PlanError, load_plan, new_plan, write_plan
from .quality import DataQuality, assess
from .report import Analysis, Options, analyse, format_report, render, to_dict
from .risk import Drawdown, Ruin, drawdown, ruin_probability
from .stats import (
    Bucket,
    Expectancy,
    block_bootstrap,
    bucket_by,
    expectancy,
    holm,
    required_n,
    t95,
)
from .verdict import Rule, Status, Verdict, decide

__version__ = "0.2.0"
__all__ = [
    "Analysis", "Bucket", "Capacity", "ColumnMap", "CostBreakdown", "DataQuality",
    "Drawdown", "Expectancy", "Options", "Plan", "PlanError", "Ruin", "Rule", "Status",
    "Trade", "UnitEconomics", "Verdict",
    "analyse", "assess", "block_bootstrap", "bucket_by", "capacity", "decide", "decompose",
    "drawdown", "expectancy", "format_report", "holm", "load", "load_plan", "new_plan",
    "parse_time", "read_rows", "render", "required_n", "ruin_probability", "t95", "to_dict",
    "unit_economics", "write_plan",
]
