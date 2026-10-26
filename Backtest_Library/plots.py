"""Reusable Matplotlib charts for quote-replay backtests.

Every public function returns a Figure. Charts never modify the result or its
metrics, and ``show=False`` makes them suitable for notebooks and batch reports.
Time-series drawing is decimated with per-bucket extrema when necessary; all
calculations still use the complete result.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
import re
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


__all__ = [
    "plot_dashboard", "plot_pnl", "plot_drawdown", "plot_trade_distribution",
    "plot_inventory", "plot_execution", "plot_markouts", "plot_win_rate",
]

_GREEN = "#14875d"
_RED = "#c84646"
_BLUE = "#2468a0"
_AMBER = "#b77b19"
_GREY = "#74818d"
_PNL_TOLERANCE = 1e-12
_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b8c1c9", "axes.labelcolor": "#263445",
    "text.color": "#263445", "xtick.color": "#526170",
    "ytick.color": "#526170", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.45,
}


def _frame(result: Any, name: str) -> pd.DataFrame:
    value = getattr(result, name, None)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _numbers(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _times(frame: pd.DataFrame) -> np.ndarray:
    source = frame["timestamp"] if "timestamp" in frame else frame.index
    return pd.to_datetime(source, utc=True).to_numpy(dtype="datetime64[ns]")


def _initial_cash(result: Any, equity: pd.DataFrame) -> float:
    config = getattr(result, "config", None)
    value = getattr(config, "initial_cash", None)
    if value is None:
        value = getattr(result, "metrics", {}).get("initial_cash")
    if value is None and len(equity):
        values = _numbers(equity, "equity", np.nan)
        pnl = _numbers(equity, "net_pnl", 0.0)
        value = values[0] - pnl[0]
    return float(value) if value is not None else 0.0


def _drawdown(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    equity = _frame(result, "equity")
    values = _numbers(equity, "equity", np.nan)
    if not len(values):
        return np.array([]), np.array([]), np.array([])
    initial = _initial_cash(result, equity)
    peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
    cash = values - peaks
    fraction = np.divide(cash, peaks, out=np.full_like(cash, np.nan), where=peaks > 0)
    return _times(equity), cash, fraction


def _decimate(
    x: np.ndarray, series: dict[str, np.ndarray], max_points: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Retain first/last and each series' local minimum/maximum per bucket."""
    if not isinstance(max_points, int) or max_points < 16:
        raise ValueError("max_points must be an integer of at least 16")
    n = len(x)
    if n <= max_points:
        return x, series
    bucket_count = max(1, (max_points - 2) // (2 * len(series) + 2))
    edges = np.linspace(0, n, bucket_count + 1, dtype=int)
    selected = {0, n - 1}
    for start, stop in zip(edges[:-1], edges[1:]):
        selected.update((int(start), int(stop - 1)))
        for values in series.values():
            chunk = values[start:stop]
            finite = np.flatnonzero(np.isfinite(chunk))
            if len(finite):
                selected.add(int(start + finite[np.argmin(chunk[finite])]))
                selected.add(int(start + finite[np.argmax(chunk[finite])]))
    indices = np.array(sorted(selected), dtype=int)
    return x[indices], {name: values[indices] for name, values in series.items()}


def _date_axis(ax: Any) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7, tz=timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    ax.set_xlabel("Time (UTC)")


def _empty(ax: Any, title: str, message: str) -> None:
    ax.set_title(title, loc="left")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=_GREY, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _closed_pnl(result: Any) -> np.ndarray:
    values = _numbers(_frame(result, "trades"), "net_pnl", np.nan)
    return values[np.isfinite(values)]


def _pnl(ax: Any, result: Any, currency: str, max_points: int) -> None:
    equity = _frame(result, "equity")
    if equity.empty:
        _empty(ax, "Cumulative PnL", "No equity observations")
        return
    net = _numbers(equity, "net_pnl", np.nan)
    if "net_pnl" not in equity:
        net = _numbers(equity, "equity", np.nan) - _initial_cash(result, equity)
    fees = _numbers(equity, "cumulative_fees")
    gross = _numbers(equity, "gross_pnl") if "gross_pnl" in equity else net + fees
    x, values = _decimate(_times(equity), {"net": net, "gross": gross, "fees": fees}, max_points)
    ax.plot(x, values["gross"], color=_BLUE, alpha=0.8, label="Gross PnL (before fees)")
    finite_net = net[np.isfinite(net)]
    net_colour = _GREEN if len(finite_net) and finite_net[-1] >= 0 else _RED
    ax.plot(x, values["net"], color=net_colour, linewidth=1.7, label="Net PnL")
    ax.plot(x, values["fees"], color=_AMBER, linestyle="--", label="Cumulative fees (cost)")
    ax.fill_between(x, 0, values["net"], where=values["net"] >= 0, color=_GREEN, alpha=0.10)
    ax.fill_between(x, 0, values["net"], where=values["net"] < 0, color=_RED, alpha=0.13)
    ax.axhline(0, color=_GREY, linewidth=0.7)
    ax.set_title("PnL before and after fees", loc="left")
    ax.set_ylabel(f"PnL / fees ({currency})")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.legend(loc="best", fontsize=8)
    _date_axis(ax)


def _underwater(ax: Any, result: Any, currency: str, max_points: int, percent: bool) -> None:
    x, cash, fraction = _drawdown(result)
    if not len(x):
        _empty(ax, "Drawdown", "No equity observations")
        return
    y = fraction * 100 if percent else cash
    x, values = _decimate(x, {"drawdown": y}, max_points)
    y = values["drawdown"]
    ax.plot(x, y, color=_RED, linewidth=1.15)
    ax.fill_between(x, y, 0, color=_RED, alpha=0.20)
    ax.axhline(0, color=_GREY, linewidth=0.7)
    ax.set_title("Drawdown from previous equity peak", loc="left")
    ax.set_ylabel("Drawdown (%)" if percent else f"Drawdown ({currency})")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    _date_axis(ax)


def _inventory(ax: Any, result: Any, max_points: int) -> None:
    equity = _frame(result, "equity")
    if equity.empty:
        _empty(ax, "Inventory", "No position observations")
        return
    x, values = _decimate(_times(equity), {"position": _numbers(equity, "position")}, max_points)
    position = values["position"]
    ax.step(x, position, where="post", color=_BLUE, linewidth=1.1)
    ax.fill_between(x, 0, position, where=position >= 0, step="post", color=_GREEN, alpha=0.13)
    ax.fill_between(x, 0, position, where=position < 0, step="post", color=_RED, alpha=0.13)
    ax.axhline(0, color=_GREY, linewidth=0.7)
    ax.set_title("Inventory held through time", loc="left")
    ax.set_ylabel("Position (base units)")
    _date_axis(ax)


def _trade_distribution(ax: Any, result: Any, currency: str, bins: int | None = None) -> None:
    pnl = _closed_pnl(result)
    if not len(pnl):
        _empty(ax, "Closed-episode PnL distribution", "No closed position episodes")
        return
    bin_count = bins if bins is not None else min(50, max(5, int(np.sqrt(len(pnl)))))
    _, edges, patches = ax.hist(pnl, bins=bin_count, edgecolor="white", linewidth=0.5)
    for left, right, patch in zip(edges[:-1], edges[1:], patches):
        patch.set_facecolor(_GREEN if (left + right) / 2 >= 0 else _RED)
        patch.set_alpha(0.85)
    ax.axvline(0, color=_GREY, linewidth=1)
    ax.axvline(float(np.mean(pnl)), color=_BLUE, linestyle="--", label=f"Mean: {np.mean(pnl):,.3f}")
    ax.set_title("Closed-episode PnL distribution", loc="left")
    ax.set_xlabel(f"Net PnL per closed episode ({currency})")
    ax.set_ylabel("Closed episodes")
    ax.legend(fontsize=8)


def _outcomes(ax: Any, result: Any) -> None:
    pnl = _closed_pnl(result)
    if not len(pnl):
        _empty(ax, "Closed position outcomes", "No closed position episodes\nWin rate: n/a")
        return
    counts = [int(np.sum(pnl > _PNL_TOLERANCE)), int(np.sum(pnl < -_PNL_TOLERANCE)),
              int(np.sum(np.abs(pnl) <= _PNL_TOLERANCE))]
    bars = ax.bar(["Win", "Loss", "Breakeven"], counts, color=[_GREEN, _RED, _GREY], width=0.6)
    for bar, count in zip(bars, counts):
        ax.annotate(f"{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_title(f"Closed position outcomes | win rate {counts[0] / len(pnl):.1%}", loc="left")
    ax.set_ylabel("Closed episodes (net of fees)")
    ax.set_ylim(0, max(counts) * 1.2 + 0.5)


def _costs(ax: Any, result: Any, currency: str) -> None:
    fills = _frame(result, "fills")
    if fills.empty:
        _empty(ax, "Execution costs", "No fills")
        return
    costs = [float(np.nansum(_numbers(fills, name)))
             for name in ("fee", "spread_cost", "slippage_cost")]
    bars = ax.bar(["Fees", "Half-spread", "Slippage"], costs, color=[_AMBER, _BLUE, _GREY], width=0.6)
    for bar, value in zip(bars, costs):
        ax.annotate(f"{value:,.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("Execution cost attribution", loc="left")
    ax.set_ylabel(f"Cost ({currency})")
    ax.margins(y=0.20)
    ax.text(0.01, 0.98, "Spread/slippage are already embedded in fill prices.",
            transform=ax.transAxes, va="top", fontsize=8, color=_GREY)


def _markout_data(result: Any) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fills = _frame(result, "fills")
    quantity = _numbers(fills, "quantity", 1.0)
    data = []
    for name in fills.columns:
        match = re.fullmatch(r"markout_bps_(\d+)", str(name))
        if match:
            values = _numbers(fills, name, np.nan)
            valid = np.isfinite(values) & np.isfinite(quantity) & (quantity > 0)
            if np.any(valid):
                data.append((int(match.group(1)), values[valid], quantity[valid]))
    return sorted(data, key=lambda item: item[0])


def _horizon_label(milliseconds: int) -> str:
    return f"{milliseconds / 1000:g} s" if milliseconds >= 1000 else f"{milliseconds} ms"


def _markouts(ax: Any, result: Any) -> None:
    data = _markout_data(result)
    if not data:
        _empty(ax, "Post-fill markouts", "No observed markout horizons\nAdd markouts before plotting this chart")
        return
    means = [float(np.average(values, weights=weights)) for _, values, weights in data]
    labels = [_horizon_label(horizon) for horizon, _, _ in data]
    bars = ax.bar(labels, means, color=[_GREEN if value >= 0 else _RED for value in means], width=0.6)
    for bar, mean, (_, values, _) in zip(bars, means, data):
        ax.annotate(f"{mean:.3f}\nn={len(values):,}",
                    (bar.get_x() + bar.get_width() / 2, mean), xytext=(0, 5 if mean >= 0 else -27),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color=_GREY, linewidth=0.8)
    ax.set_title("Post-fill markout | positive is favorable", loc="left")
    ax.set_ylabel("Quantity-weighted markout (bps)")
    ax.set_xlabel("Time after fill")
    ax.margins(y=0.35)


def _finish(fig: Figure, save_path: str | Path | None, show: bool) -> Figure:
    if save_path is not None:
        path = Path(save_path)
        if not path.suffix:
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _metric_text(value: Any, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(number):
        return "n/a"
    return f"{number:.2%}" if percent else f"{number:,.2f}"


def plot_dashboard(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    currency: str = "USDT", max_points: int = 10_000,
) -> Figure:
    """Six-panel overview of PnL, risk, inventory and closed-trade outcomes.

    The last panel shows quantity-weighted markouts when available, otherwise
    execution-cost attribution. Win rate uses completed flat-to-flat episodes,
    including breakevens in its denominator. Open inventory remains in MTM PnL.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(16, 13))
        _pnl(axes[0, 0], result, currency, max_points)
        _underwater(axes[0, 1], result, currency, max_points, True)
        _inventory(axes[1, 0], result, max_points)
        _trade_distribution(axes[1, 1], result, currency)
        _outcomes(axes[2, 0], result)
        if _markout_data(result):
            _markouts(axes[2, 1], result)
        else:
            _costs(axes[2, 1], result, currency)
        metrics = getattr(result, "metrics", {}) or {}
        pnl = _closed_pnl(result)
        win_rate = np.mean(pnl > _PNL_TOLERANCE) if len(pnl) else None
        sharpe = metrics.get("daily_sharpe", metrics.get("sharpe_daily"))
        summary = (
            f"Net PnL: {_metric_text(metrics.get('net_pnl'))} {currency}   |   "
            f"Return: {_metric_text(metrics.get('total_return'), percent=True)}   |   "
            f"Daily Sharpe: {_metric_text(sharpe)}   |   "
            f"Closed episodes: {len(pnl):,}   |   Win rate: {_metric_text(win_rate, percent=True)}"
        )
        fig.suptitle("Backtest performance", x=0.06, y=0.995, ha="left", fontsize=19, fontweight="bold")
        fig.text(0.06, 0.965, summary, fontsize=10)
        fig.text(0.06, 0.012, "UTC timestamps. Plot decimation preserves local extrema; metrics use all observations.",
                 fontsize=8, color=_GREY)
        fig.tight_layout(rect=(0.025, 0.03, 0.995, 0.945), h_pad=2.7, w_pad=2.5)
        return _finish(fig, save_path, show)


def plot_pnl(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    currency: str = "USDT", max_points: int = 10_000,
) -> Figure:
    """Plot cumulative gross PnL, net PnL and fees; embedded costs are not deducted twice."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        _pnl(ax, result, currency, max_points)
        fig.tight_layout()
        return _finish(fig, save_path, show)


def plot_drawdown(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    currency: str = "USDT", percent: bool = True, max_points: int = 10_000,
) -> Figure:
    """Plot equity drawdown against running peaks, including initial capital."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _underwater(ax, result, currency, max_points, percent)
        fig.tight_layout()
        return _finish(fig, save_path, show)


def plot_inventory(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    max_points: int = 10_000,
) -> Figure:
    """Plot held inventory with long and short exposure shaded separately."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _inventory(ax, result, max_points)
        fig.tight_layout()
        return _finish(fig, save_path, show)


def plot_trade_distribution(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    currency: str = "USDT", bins: int | None = None,
) -> Figure:
    """Show net closed-episode PnL distribution and win/loss/breakeven counts."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        _trade_distribution(axes[0], result, currency, bins)
        _outcomes(axes[1], result)
        fig.tight_layout(w_pad=3)
        return _finish(fig, save_path, show)


def plot_win_rate(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
) -> Figure:
    """Plot closed-episode win/loss/breakeven counts and net-of-fees win rate."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        _outcomes(ax, result)
        fig.tight_layout()
        return _finish(fig, save_path, show)


def plot_execution(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
    currency: str = "USDT",
) -> Figure:
    """Plot costs, realized latency, IOC order outcomes and effective spread.

    Effective spread is ``2 * side * (fill_price - mid_price) / mid_price``
    in basis points, with positive values denoting execution cost. Latency is
    observed decision-to-fill time, including quote-grid granularity.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        _costs(axes[0, 0], result, currency)
        fills, orders = _frame(result, "fills"), _frame(result, "orders")
        latency = _numbers(fills, "latency_ms", np.nan)
        latency = latency[np.isfinite(latency)]
        if len(latency):
            axes[0, 1].hist(latency, bins=min(40, max(5, int(np.sqrt(len(latency))))),
                            color=_BLUE, alpha=0.8, edgecolor="white")
            axes[0, 1].set_title("Realized execution latency", loc="left")
            axes[0, 1].set_xlabel("Decision to fill (ms)")
            axes[0, 1].set_ylabel("Fills")
            axes[0, 1].ticklabel_format(axis="x", style="plain", useOffset=False)
        else:
            _empty(axes[0, 1], "Realized execution latency", "No fills")
        if len(orders) and "status" in orders:
            counts = orders["status"].astype(str).value_counts()
            preferred = [status for status in ("filled", "partial", "unfilled", "expired") if status in counts]
            preferred += [status for status in counts.index if status not in preferred]
            colours = {"filled": _GREEN, "partial": _AMBER, "unfilled": _GREY, "expired": _RED}
            axes[1, 0].bar(preferred, [counts[status] for status in preferred],
                           color=[colours.get(status, _GREY) for status in preferred])
            axes[1, 0].set_title("Order outcomes", loc="left")
            axes[1, 0].set_ylabel("Orders")
        else:
            _empty(axes[1, 0], "Order outcomes", "No submitted orders")
        if len(fills) and "side" in fills:
            side = np.where(fills["side"].astype(str).str.lower().to_numpy() == "buy", 1.0, -1.0)
            price, mid = _numbers(fills, "price", np.nan), _numbers(fills, "mid_price", np.nan)
            spread = np.divide(2 * side * (price - mid), mid, out=np.full(len(fills), np.nan), where=mid > 0) * 10_000
            finite = np.isfinite(spread)
            quantity = _numbers(fills, "quantity", 1.0)[finite]
            spread = spread[finite]
        else:
            spread = np.array([])
        if len(spread):
            _, edges, _ = axes[1, 1].hist(
                spread, bins=min(40, max(5, int(np.sqrt(len(spread))))),
                color=_AMBER, alpha=0.85, edgecolor="white",
            )
            padding = (edges[-1] - edges[0]) * 0.05
            lower, upper = edges[0] - padding, edges[-1] + padding
            # A distant zero line would otherwise flatten a tightly clustered histogram.
            if lower <= 0 <= upper:
                axes[1, 1].axvline(0, color=_GREY, linewidth=0.8)
            valid_quantity = np.isfinite(quantity) & (quantity > 0)
            mean_spread = (float(np.average(spread[valid_quantity], weights=quantity[valid_quantity]))
                           if np.any(valid_quantity) else float(np.mean(spread)))
            axes[1, 1].axvline(mean_spread, color=_BLUE, linewidth=1.15, linestyle="--")
            axes[1, 1].set_xlim(lower, upper)
            axes[1, 1].ticklabel_format(axis="x", style="plain", useOffset=False)
            axes[1, 1].set_title("Effective spread paid", loc="left")
            axes[1, 1].set_xlabel("Effective spread (bps)")
            axes[1, 1].set_ylabel("Fills")
            axes[1, 1].text(
                0.02, 0.98, f"Qty-weighted mean: {mean_spread:.6g} bps",
                transform=axes[1, 1].transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        else:
            _empty(axes[1, 1], "Effective spread paid", "No valid fills")
        fig.tight_layout(h_pad=3, w_pad=3)
        return _finish(fig, save_path, show)


def plot_markouts(
    result: Any, *, save_path: str | Path | None = None, show: bool = False,
) -> Figure:
    """Plot quantity-weighted mean markouts and the per-fill distributions.

    Markouts are measured from execution price to a later mid-price, signed
    for the trade side. They include entry spread/slippage and exclude fees.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _markouts(axes[0], result)
        data = _markout_data(result)
        if data:
            boxes = axes[1].boxplot([values for _, values, _ in data], patch_artist=True,
                                   showfliers=False)
            for patch in boxes["boxes"]:
                patch.set_facecolor(_BLUE)
                patch.set_alpha(0.25)
            axes[1].set_xticks(range(1, len(data) + 1), [_horizon_label(horizon) for horizon, _, _ in data])
            axes[1].axhline(0, color=_GREY, linewidth=0.8)
            axes[1].set_title("Per-fill markout distribution", loc="left")
            axes[1].set_xlabel("Time after fill")
            axes[1].set_ylabel("Signed markout (bps; unweighted)")
            axes[1].text(0.01, 0.98, "Boxes: median and interquartile range; outliers hidden.",
                         transform=axes[1].transAxes, va="top", fontsize=8, color=_GREY)
        else:
            _empty(axes[1], "Per-fill markout distribution", "No observed markout horizons")
        fig.tight_layout(w_pad=3)
        return _finish(fig, save_path, show)
