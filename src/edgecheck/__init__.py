"""edgecheck -- does your strategy's claimed edge survive contact with reality?

Point it at a file of settled trades and it answers, in order:

  1. Do costs exceed the gross edge?   (if yes, nothing else matters)
  2. Is the result distinguishable from noise?
  3. Where is the model's probability wrong?
  4. Is the loss concentrated in one slice?
  5. How much more data would settle it?
"""

from .fees import CostBreakdown, UnitEconomics, decompose, unit_economics
from .loaders import ColumnMap, Trade, load, read_rows
from .report import render
from .stats import Bucket, Expectancy, bucket_by, expectancy, required_n, t95

__version__ = "0.1.0"
__all__ = [
    "Bucket", "ColumnMap", "CostBreakdown", "Expectancy", "Trade",
    "UnitEconomics", "bucket_by", "decompose", "expectancy", "load",
    "read_rows", "render", "required_n", "t95", "unit_economics",
]
