"""Reusable local quote backtests, HFT diagnostics and plotting functions."""

from .models import BacktestConfig, BacktestResult
from .engine import run_backtest
from .metrics import add_markouts, compute_metrics
from .plots import (
    plot_dashboard, plot_drawdown, plot_execution, plot_inventory,
    plot_markouts, plot_pnl, plot_trade_distribution, plot_win_rate,
)

__version__ = "0.1.0"
__all__ = [
    "BacktestConfig", "BacktestResult", "run_backtest", "compute_metrics", "add_markouts",
    "plot_dashboard", "plot_pnl", "plot_drawdown", "plot_inventory", "plot_execution",
    "plot_markouts", "plot_trade_distribution", "plot_win_rate",
]
