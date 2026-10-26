"""Run an illustrative imbalance rule on two hours of our saved LOB data.

From the HFT root: python Backtest_Library/examples/run_demo.py
Requires the optional Polars dependency and the existing LOB Parquet file.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from Backtest_Library import (
    BacktestConfig, run_backtest, plot_dashboard, plot_pnl, plot_drawdown,
    plot_inventory, plot_trade_distribution, plot_execution, plot_markouts,
)


def make_demo_result(rows=7200):
    """Return a real-data plumbing example, not a claim of profitable alpha."""
    path = PROJECT_ROOT / "LOB" / "processed" / "btcusdt_lob_1s.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Prepare the LOB dataset first: {path}")
    quotes = pl.scan_parquet(path).head(rows).collect()
    bid_qty = quotes["bid_qty"].to_numpy()
    ask_qty = quotes["ask_qty"].to_numpy()
    imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    target = np.where(imbalance > 0.30, 0.001,
                      np.where(imbalance < -0.30, -0.001, 0.0))
    # Give ordinary next-quote orders time to flatten before the data ends.
    target[-5:] = 0.0
    result = run_backtest(quotes, target, config=BacktestConfig(
        initial_cash=10_000, fee_bps=5.0, slippage_bps=0.1,
        latency_ms=100, max_abs_position=0.001, max_quote_fraction=0.10,
    ))
    return quotes, target, result


def save_demo_figures(result, directory):
    """Write reusable examples of every supported chart."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    charts = {
        "dashboard": plot_dashboard, "pnl": plot_pnl, "drawdown": plot_drawdown,
        "inventory": plot_inventory, "trade_distribution": plot_trade_distribution,
        "execution": plot_execution, "markouts": plot_markouts,
    }
    from Backtest_Library.plots import plot_win_rate
    charts["win_rate"] = plot_win_rate
    paths = {}
    for name, plotter in charts.items():
        paths[name] = directory / f"{name}.png"
        figure = plotter(result, save_path=paths[name])
        plt.close(figure)
    result.save(directory / "report")
    return paths


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    quotes, target, result = make_demo_result()
    output = Path(__file__).resolve().parent / "output"
    paths = save_demo_figures(result, output)
    print(f"Quotes: {len(quotes):,}; closed episodes: {result.metrics['closed_trades']:,}")
    print(result.summary().to_string())
    print(f"Saved {len(paths)} charts and ledgers to {output}")
