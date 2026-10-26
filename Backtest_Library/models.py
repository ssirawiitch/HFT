"""Public configuration and results for the local quote-replay backtester."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Quantities are base-asset units; cash/PnL use the quote currency.

    The default fee is an illustrative assumption, not an exchange fee schedule.
    Orders always execute on a strictly later quote, even with zero latency.
    """

    initial_cash: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    max_abs_position: float = 1.0
    max_quote_fraction: float = 1.0
    max_quote_age_ms: float = 2000.0
    max_quote_delay_ms: float = 2000.0
    close_final_position: bool = False
    markout_horizons_ms: tuple[int, ...] = (1000, 5000, 10000)
    annualization_days: float = 365.25

    def __post_init__(self) -> None:
        positive = ("initial_cash", "annualization_days")
        nonnegative = (
            "fee_bps", "slippage_bps", "latency_ms", "max_abs_position",
            "max_quote_age_ms", "max_quote_delay_ms",
        )
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name in positive else 'nonnegative'}")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be below 10000")
        if not math.isfinite(self.max_quote_fraction) or not 0 <= self.max_quote_fraction <= 1:
            raise ValueError("max_quote_fraction must be between 0 and 1")
        if any(not isinstance(h, int) or isinstance(h, bool) or h <= 0 for h in self.markout_horizons_ms):
            raise ValueError("markout_horizons_ms must contain positive integer milliseconds")


@dataclass
class BacktestResult:
    """Materialized ledgers, performance metrics and reproducible assumptions."""

    equity: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict
    config: BacktestConfig

    def summary(self) -> pd.Series:
        """Return metrics as a notebook-friendly Series; ratios are fractions."""
        return pd.Series(self.metrics, name="value", dtype=object)

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Export the ledgers as CSV and configuration/metrics as JSON."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("equity", "fills", "trades", "orders"):
            path = directory / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            paths[name] = path
        paths["summary"] = directory / "summary.json"
        paths["summary"].write_text(
            json.dumps({"config": asdict(self.config), "metrics": self.metrics}, indent=2,
                       allow_nan=False, default=lambda value: value.item()) + "\n",
            encoding="utf-8",
        )
        return paths
