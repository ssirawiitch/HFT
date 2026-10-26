"""Performance and execution-quality measurements for quote-replay backtests.

Money is in the account's quote currency, positions are asset units, basis-point
measurements use 10,000 bps per unit return, and percentage metrics are fractions.
Undefined ratios return ``None``. Spread/slippage are already embedded in fill
prices; their cost summaries must not be subtracted from PnL a second time.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


_EPSILON = 1e-12


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _numbers(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains non-finite values")
    return values


def _timestamps(frame: pd.DataFrame, *, ordered: bool) -> pd.DatetimeIndex:
    # Normalize the unit explicitly: pandas can preserve microsecond resolution.
    values = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).astype(
        "datetime64[ns, UTC]"
    )
    if values.hasnans:
        raise ValueError("timestamp contains missing values")
    if ordered and (not values.is_monotonic_increasing or values.has_duplicates):
        raise ValueError("timestamp must be increasing and unique")
    return values


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    return float(np.average(values[valid], weights=weights[valid]))


def _last_or_sum(
    equity: pd.DataFrame, cumulative_column: str, fills: pd.DataFrame, fill_column: str
) -> float:
    if cumulative_column in equity:
        return float(_numbers(equity, cumulative_column)[-1])
    return float(_numbers(fills, fill_column).sum()) if fill_column in fills else 0.0


def _daily_returns(
    timestamps: pd.DatetimeIndex, equity_values: np.ndarray, initial_cash: float
) -> pd.Series:
    """Observed UTC-day close returns; do not label multi-day gaps as one day."""
    curve = pd.Series(equity_values, index=timestamps)
    closes = curve.resample("1D").last().dropna()
    daily = closes.pct_change(fill_method=None)
    if not closes.empty:
        daily.iloc[0] = closes.iloc[0] / initial_cash - 1.0
    if len(closes) > 1:
        consecutive = np.diff(closes.index.asi8) == 86_400_000_000_000
        daily.iloc[1:] = daily.iloc[1:].where(consecutive)
    return daily.replace([np.inf, -np.inf], np.nan).dropna()


def compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    fills: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
    *,
    initial_cash: float | None = None,
    annualization_days: float = 365.25,
) -> dict[str, float | int | None]:
    """Measure account, closed-episode, execution, and inventory performance.

    ``equity`` needs ``timestamp`` and ``equity``; the engine supplies the other
    account columns. ``trades`` must contain *closed flat-to-flat episodes*:
    win rate uses positive net episode PnL divided by all closed episodes,
    including breakeven episodes. Open inventory remains in account MTM PnL.

    Sharpe/Sortino use observed UTC-day end returns and a zero risk-free/target
    return, annualized by ``sqrt(annualization_days)``. The first observed day
    uses initial capital; partial boundary days are included. Fewer than two
    daily returns, nonpositive account equity, or a zero denominator gives
    ``None``. This conventional scaling is not an autocorrelation correction.

    Inventory exposure is time weighted using actual timestamp gaps, with the
    last position held over each following interval; the final row has zero
    duration. Effective spread and markout means are quantity weighted.
    """
    _require_columns(equity, ("timestamp", "equity"), "equity")
    if equity.empty:
        raise ValueError("equity must contain at least one row")
    for name, frame in (("trades", trades), ("fills", fills), ("orders", orders)):
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame or None")
    if not math.isfinite(annualization_days) or annualization_days <= 0:
        raise ValueError("annualization_days must be finite and positive")

    trades_provided = trades is not None
    trades = pd.DataFrame() if trades is None else trades
    fills = pd.DataFrame() if fills is None else fills
    orders_provided = orders is not None
    orders = pd.DataFrame() if orders is None else orders
    timestamps = _timestamps(equity, ordered=True)
    equity_values = _numbers(equity, "equity")
    if initial_cash is None:
        initial_cash = float(equity_values[0])
        if "net_pnl" in equity:
            initial_cash -= float(_numbers(equity, "net_pnl")[0])
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    initial_cash = float(initial_cash)

    final_equity = float(equity_values[-1])
    net_pnl = final_equity - initial_cash
    fees = _last_or_sum(equity, "cumulative_fees", fills, "fee")
    spread_cost = _last_or_sum(equity, "cumulative_spread_cost", fills, "spread_cost")
    slippage_cost = _last_or_sum(
        equity, "cumulative_slippage_cost", fills, "slippage_cost"
    )
    gross_pnl = (
        float(_numbers(equity, "gross_pnl")[-1])
        if "gross_pnl" in equity else net_pnl + fees
    )
    high_water = np.maximum.accumulate(np.r_[initial_cash, equity_values])[1:]
    drawdown = high_water - equity_values
    elapsed_seconds = float((timestamps[-1] - timestamps[0]).total_seconds())
    daily_returns = _daily_returns(timestamps, equity_values, initial_cash)
    daily_sharpe = daily_sortino = annualized_volatility = None
    if len(daily_returns) >= 2 and (equity_values > 0).all():
        returns = daily_returns.to_numpy(dtype=float)
        deviation = float(returns.std(ddof=1))
        annualized_volatility = deviation * math.sqrt(annualization_days)
        if deviation > _EPSILON:
            daily_sharpe = float(returns.mean() / deviation * math.sqrt(annualization_days))
        downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        if downside > _EPSILON:
            daily_sortino = float(returns.mean() / downside * math.sqrt(annualization_days))

    result: dict[str, float | int | None] = {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "total_return": net_pnl / initial_cash,
        "total_fees": fees,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "max_drawdown": float(drawdown.max()),
        "max_drawdown_pct": float((drawdown / high_water).max()),
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "annualized_volatility": annualized_volatility,
        "daily_return_count": int(len(daily_returns)),
        "observed_days": int(timestamps.normalize().nunique()),
        "elapsed_seconds": elapsed_seconds,
    }

    if not trades.empty:
        _require_columns(trades, ("net_pnl",), "trades")
        trade_pnl = _numbers(trades, "net_pnl")
    else:
        trade_pnl = np.array([], dtype=float)
    winning = trade_pnl[trade_pnl > _EPSILON]
    losing = trade_pnl[trade_pnl < -_EPSILON]
    gross_profit = float(winning.sum())
    gross_loss = float(-losing.sum())
    closed_count = len(trade_pnl)
    closed_pnl = float(trade_pnl.sum()) if trades_provided else None
    average_win = float(winning.mean()) if winning.size else None
    average_loss = float(losing.mean()) if losing.size else None
    result.update({
        "closed_trades": closed_count,
        "winning_trades": int(winning.size),
        "losing_trades": int(losing.size),
        "breakeven_trades": int(closed_count - winning.size - losing.size),
        "win_rate": float(winning.size / closed_count) if closed_count else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > _EPSILON else None,
        "gross_closed_profit": gross_profit,
        "gross_closed_loss": gross_loss,
        "avg_win": average_win,
        "avg_loss": average_loss,
        "payoff_ratio": average_win / abs(average_loss)
        if average_win is not None and average_loss is not None else None,
        "expectancy": float(trade_pnl.mean()) if closed_count else None,
        "closed_trade_net_pnl": closed_pnl,
        "open_trade_pnl": net_pnl - closed_pnl if closed_pnl is not None else None,
        "avg_holding_seconds": float(_numbers(trades, "holding_seconds").mean())
        if closed_count and "holding_seconds" in trades else None,
        "median_holding_seconds": float(np.median(_numbers(trades, "holding_seconds")))
        if closed_count and "holding_seconds" in trades else None,
    })

    quantities = np.array([], dtype=float)
    turnover = 0.0
    effective_spread = None
    if not fills.empty:
        _require_columns(fills, ("quantity", "price"), "fills")
        quantities = _numbers(fills, "quantity")
        prices = _numbers(fills, "price")
        if (quantities <= 0).any() or (prices <= 0).any():
            raise ValueError("fill quantities and prices must be positive")
        turnover = float(np.dot(quantities, prices))
        if {"mid_price", "side"}.issubset(fills.columns):
            mids = _numbers(fills, "mid_price")
            if (mids <= 0).any():
                raise ValueError("fill mid_price must be positive")
            side = fills["side"].astype(str).str.lower()
            if not side.isin(("buy", "sell")).all():
                raise ValueError("fill side must be buy or sell")
            signs = np.where(side.eq("buy"), 1.0, -1.0)
            effective_spread = _weighted_mean(
                2.0 * signs * (prices - mids) / mids * 10_000.0, quantities
            )
    result.update({
        "fills": int(len(fills)),
        "traded_quantity": float(quantities.sum()),
        "turnover_notional": turnover,
        "turnover_ratio": turnover / initial_cash,
        "net_edge_bps": net_pnl / turnover * 10_000.0 if turnover > 0 else None,
        "net_pnl_per_million_traded": net_pnl / turnover * 1_000_000.0
        if turnover > 0 else None,
        "effective_spread_bps": effective_spread,
        "mean_execution_latency_ms": float(_numbers(fills, "latency_ms").mean())
        if not fills.empty and "latency_ms" in fills else None,
        "fills_per_day": len(fills) * 86_400.0 / elapsed_seconds
        if elapsed_seconds > 0 else None,
        "closed_trades_per_day": closed_count * 86_400.0 / elapsed_seconds
        if elapsed_seconds > 0 else None,
    })
    for column in fills.columns:
        if column.startswith("markout_bps_"):
            values = pd.to_numeric(fills[column], errors="raise").to_numpy(dtype=float)
            result[column] = _weighted_mean(values, quantities) if len(fills) else None
            result[column.replace("markout_bps_", "markout_coverage_")] = (
                float(np.isfinite(values).mean()) if len(fills) else None
            )

    requested = filled = np.array([], dtype=float)
    if not orders.empty:
        _require_columns(orders, ("requested_qty", "filled_qty"), "orders")
        requested, filled = _numbers(orders, "requested_qty"), _numbers(orders, "filled_qty")
        if (requested < 0).any() or (filled < 0).any() or (filled - requested > _EPSILON).any():
            raise ValueError("order quantities must satisfy 0 <= filled_qty <= requested_qty")
    requested_total = float(requested.sum())
    result.update({
        "submitted_orders": int(len(orders)),
        "filled_orders": int(np.count_nonzero(filled > _EPSILON)) if orders_provided else None,
        "fill_rate": float(np.count_nonzero(filled > _EPSILON) / len(orders))
        if len(orders) else None,
        "quantity_fill_ratio": float(filled.sum() / requested_total)
        if requested_total > 0 else None,
    })

    if "position" in equity:
        positions = _numbers(equity, "position")
        durations = np.diff(timestamps.asi8).astype(float) / 1_000_000_000.0
        exposure_seconds = float(np.dot(np.abs(positions[:-1]) > _EPSILON, durations))
        result.update({
            "max_abs_inventory": float(np.abs(positions).max()),
            "mean_abs_inventory": float(np.dot(np.abs(positions[:-1]), durations) / elapsed_seconds)
            if elapsed_seconds > 0 else None,
            "mean_inventory": float(np.dot(positions[:-1], durations) / elapsed_seconds)
            if elapsed_seconds > 0 else None,
            "inventory_rms": float(np.sqrt(np.dot(positions[:-1] ** 2, durations) / elapsed_seconds))
            if elapsed_seconds > 0 else None,
            "time_in_market": exposure_seconds / elapsed_seconds if elapsed_seconds > 0 else None,
            "exposure_seconds": exposure_seconds,
            "final_position": float(positions[-1]),
        })
    else:
        result.update({key: None for key in (
            "max_abs_inventory", "mean_abs_inventory", "mean_inventory", "inventory_rms",
            "time_in_market", "exposure_seconds", "final_position"
        )})
    return result


def add_markouts(
    fills: pd.DataFrame,
    quotes: pd.DataFrame,
    horizons_ms: Iterable[int] = (1000, 5000, 10000),
    max_quote_age_ms: float = 2000,
) -> pd.DataFrame:
    """Append side-signed, fee-excluding future-mid markouts in basis points.

    For a buy, markout is ``(future_mid - fill_price) / fill_price * 1e4``;
    for a sell its sign is reversed. Positive means a favorable price move.
    Use the latest quote *at or before* ``fill_time + horizon``. A horizon
    outside the observed quote range, or with a quote older than the limit,
    gets NaN instead of extrapolation. If quotes contain ``book_age_ms``, include
    that source age as well as the time since the sampled quote. Quotes need timestamp and mid_price,
    or timestamp plus best_bid/best_ask. All input frames remain unchanged.
    """
    if not isinstance(fills, pd.DataFrame) or not isinstance(quotes, pd.DataFrame):
        raise TypeError("fills and quotes must be pandas DataFrames")
    horizons: list[int] = []
    for horizon in horizons_ms:
        value = int(horizon)
        if value != horizon or value < 0 or value > np.iinfo(np.int64).max // 1_000_000:
            raise ValueError("horizons_ms must contain nonnegative integer milliseconds")
        if value not in horizons:
            horizons.append(value)
    if not math.isfinite(max_quote_age_ms) or max_quote_age_ms < 0:
        raise ValueError("max_quote_age_ms must be finite and nonnegative")
    output = fills.copy()
    for horizon in horizons:
        output[f"markout_bps_{horizon}"] = np.nan
    if fills.empty or not horizons:
        return output

    _require_columns(fills, ("timestamp", "price", "side"), "fills")
    fill_times = _timestamps(fills, ordered=False).asi8
    prices = _numbers(fills, "price")
    if (prices <= 0).any():
        raise ValueError("fill prices must be positive")
    sides = fills["side"].astype(str).str.lower()
    if not sides.isin(("buy", "sell")).all():
        raise ValueError("fill side must be buy or sell")
    signs = np.where(sides.eq("buy"), 1.0, -1.0)
    if quotes.empty:
        return output
    _require_columns(quotes, ("timestamp",), "quotes")
    quote_times = _timestamps(quotes, ordered=True).asi8
    if "mid_price" in quotes:
        mids = _numbers(quotes, "mid_price")
    else:
        _require_columns(quotes, ("best_bid", "best_ask"), "quotes")
        mids = (_numbers(quotes, "best_bid") + _numbers(quotes, "best_ask")) / 2
    if (mids <= 0).any():
        raise ValueError("quote mid prices must be positive")
    book_age_ms = (
        _numbers(quotes, "book_age_ms") if "book_age_ms" in quotes
        else np.zeros(len(quotes), dtype=float)
    )
    if (book_age_ms < 0).any():
        raise ValueError("book_age_ms cannot be negative")

    for horizon in horizons:
        horizon_ns = horizon * 1_000_000
        safe_add = fill_times <= np.iinfo(np.int64).max - horizon_ns
        targets = np.zeros_like(fill_times)
        targets[safe_add] = fill_times[safe_add] + horizon_ns
        indices = np.searchsorted(quote_times, targets, side="right") - 1
        usable = safe_add & (indices >= 0) & (targets <= quote_times[-1])
        safe_indices = np.clip(indices, 0, len(quote_times) - 1)
        age_ns = targets - quote_times[safe_indices] + book_age_ms[safe_indices] * 1_000_000
        usable &= (age_ns >= 0) & (age_ns <= max_quote_age_ms * 1_000_000)
        values = np.full(len(fills), np.nan, dtype=float)
        values[usable] = (
            signs[usable] * (mids[safe_indices[usable]] - prices[usable])
            / prices[usable] * 10_000.0
        )
        output[f"markout_bps_{horizon}"] = values
    return output


__all__ = ["compute_metrics", "add_markouts"]
