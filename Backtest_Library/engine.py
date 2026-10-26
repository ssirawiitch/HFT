"""Own implementation of a small-order, sampled-quote IOC taker simulator.

It models one outstanding order, next-quote execution, displayed top-level
capacity, fees and adverse slippage. It does not model passive limit-order queues,
market impact, funding, margin or exchange connectivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import add_markouts, compute_metrics
from .models import BacktestConfig, BacktestResult

EPS = 1e-12
FILL_COLUMNS = [
    "timestamp", "decision_time", "order_id", "side", "quantity", "price",
    "mid_price", "fee", "spread_cost", "slippage_cost", "latency_ms", "is_liquidation",
]
TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price",
    "gross_pnl", "fees", "net_pnl", "holding_seconds",
]
ORDER_COLUMNS = [
    "order_id", "decision_time", "arrival_time", "side", "requested_qty",
    "filled_qty", "status", "is_liquidation", "reason",
]


def _column(data: Any, name: str) -> np.ndarray:
    column = data[name]
    return column.to_numpy() if hasattr(column, "to_numpy") else np.asarray(column)


def _prepare_quotes(data: Any) -> pd.DataFrame:
    """Accept pandas/Polars DataFrames without requiring PyArrow."""
    if not hasattr(data, "columns") or not hasattr(data, "__getitem__"):
        raise TypeError("quotes must be a materialized pandas or Polars DataFrame")
    timestamp = "timestamp_utc" if "timestamp_utc" in data.columns else "timestamp"
    required = [timestamp, "best_bid", "bid_qty", "best_ask", "ask_qty"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing quote columns: {sorted(missing)}")
    columns = required + (["book_age_ms"] if "book_age_ms" in data.columns else [])
    frame = pd.DataFrame({name: _column(data, name) for name in columns})
    if frame.empty:
        raise ValueError("quotes cannot be empty")
    frame = frame.rename(columns={timestamp: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    if frame["timestamp"].isna().any():
        raise ValueError("Quote timestamps cannot be missing")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Quote timestamps must be sorted and unique; align targets before calling")
    for name in ("best_bid", "bid_qty", "best_ask", "ask_qty"):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"{name} must contain finite numbers")
    if (frame[["bid_qty", "ask_qty"]] < 0).any().any():
        raise ValueError("Quote quantities cannot be negative")
    if (frame["best_bid"] <= 0).any() or (frame["best_bid"] >= frame["best_ask"]).any():
        raise ValueError("Quotes require 0 < best_bid < best_ask")
    if "book_age_ms" not in frame:
        frame["book_age_ms"] = 0.0
    frame["book_age_ms"] = pd.to_numeric(frame["book_age_ms"], errors="raise").astype(float)
    if not np.isfinite(frame["book_age_ms"].to_numpy()).all() or (frame["book_age_ms"] < 0).any():
        raise ValueError("book_age_ms must contain finite nonnegative values")
    frame["mid_price"] = (frame["best_bid"] + frame["best_ask"]) / 2
    return frame


class _Episodes:
    """Aggregate fills into flat-to-flat episodes, splitting reversal fees."""

    def __init__(self) -> None:
        self.current: dict | None = None
        self.closed: list[dict] = []

    def _open(self, timestamp, direction, quantity, price, fee) -> None:
        if self.current is None:
            self.current = {"entry_time": timestamp, "direction": direction,
                            "entry_qty": 0.0, "entry_notional": 0.0,
                            "exit_qty": 0.0, "exit_notional": 0.0, "fees": 0.0}
        self.current["entry_qty"] += quantity
        self.current["entry_notional"] += quantity * price
        self.current["fees"] += fee

    def record(self, timestamp, signed_quantity, price, fee, position_before) -> None:
        quantity = abs(signed_quantity)
        direction = 1 if signed_quantity > 0 else -1
        if abs(position_before) <= EPS or direction * position_before > 0:
            self._open(timestamp, direction, quantity, price, fee)
            return
        closing_quantity = min(quantity, abs(position_before))
        closing_fee = fee * closing_quantity / quantity
        episode = self.current
        if episode is None:
            raise RuntimeError("Episode accounting lost an open position")
        episode["exit_qty"] += closing_quantity
        episode["exit_notional"] += closing_quantity * price
        episode["fees"] += closing_fee
        if closing_quantity >= abs(position_before) - EPS:
            gross = episode["direction"] * (episode["exit_notional"] - episode["entry_notional"])
            self.closed.append({
                "entry_time": episode["entry_time"], "exit_time": timestamp,
                "side": "long" if episode["direction"] > 0 else "short",
                "quantity": episode["entry_qty"],
                "entry_price": episode["entry_notional"] / episode["entry_qty"],
                "exit_price": episode["exit_notional"] / episode["exit_qty"],
                "gross_pnl": gross, "fees": episode["fees"],
                "net_pnl": gross - episode["fees"],
                "holding_seconds": (timestamp - episode["entry_time"]).total_seconds(),
            })
            self.current = None
        remainder = quantity - closing_quantity
        if remainder > EPS:
            self._open(timestamp, direction, remainder, price, fee - closing_fee)


def run_backtest(
    quotes: Any,
    target_positions: Any = None,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest desired base-asset positions against future executable quotes.

    ``target_positions`` is a column name or one numeric target per quote, e.g.
    +0.01 BTC, 0, -0.01 BTC. Targets are clipped to ``max_abs_position``.
    There is one in-flight IOC order. Target changes wait until it resolves;
    partial remainders cancel, and later decisions may submit the residual.

    Zero configured latency still uses the next strictly later quote. Actual
    fills cannot have finer time resolution than the input. Optional terminal
    liquidation is an explicit same-final-quote convention, remains depth-capped,
    and can leave residual inventory. PnL is always marked to the quote midpoint.
    """
    config = config or BacktestConfig()
    if target_positions is None:
        target_positions = "target_position"
    if isinstance(target_positions, str):
        if target_positions not in quotes.columns:
            raise ValueError(f"Target column not found: {target_positions}")
        targets = _column(quotes, target_positions)
    else:
        targets = np.asarray(target_positions)
    frame = _prepare_quotes(quotes)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1 or len(targets) != len(frame):
        raise ValueError("target_positions must contain one value per quote")
    if not np.isfinite(targets).all():
        raise ValueError("target_positions must contain finite numbers")
    targets = np.clip(targets, -config.max_abs_position, config.max_abs_position)

    times = frame["timestamp"].array
    time_ns = times.asi8
    bid = frame["best_bid"].to_numpy()
    ask = frame["best_ask"].to_numpy()
    bid_size = frame["bid_qty"].to_numpy() * config.max_quote_fraction
    ask_size = frame["ask_qty"].to_numpy() * config.max_quote_fraction
    age = frame["book_age_ms"].to_numpy()
    mid = frame["mid_price"].to_numpy()
    count = len(frame)
    cash, position = config.initial_cash, 0.0
    cumulative_fees = cumulative_spread = cumulative_slippage = 0.0
    fills, orders = [], []
    # Avoid millions of per-row dictionaries for month-long quote histories.
    history_columns = [
        "position", "cash", "equity", "net_pnl", "gross_pnl",
        "cumulative_fees", "cumulative_spread_cost", "cumulative_slippage_cost",
    ]
    history = np.empty((count, len(history_columns)), dtype=float)
    pending = None
    episodes = _Episodes()
    latency_ns = int(round(config.latency_ms * 1_000_000))

    def make_order(i, quantity, liquidation=False):
        due = int(time_ns[i]) + (0 if liquidation else latency_ns)
        arrival = i if liquidation else max(i + 1, int(np.searchsorted(time_ns, due, side="left")))
        order = {
            "order_id": len(orders), "decision_time": times[i],
            "arrival_time": times[arrival] if arrival < count else pd.NaT,
            "side": "buy" if quantity > 0 else "sell",
            "requested_qty": abs(quantity), "filled_qty": 0.0,
            "status": "pending", "is_liquidation": liquidation, "reason": "",
            "_quantity": quantity, "_arrival": arrival, "_due": due,
        }
        orders.append(order)
        return order

    def execute(order, i, remaining):
        nonlocal cash, position, cumulative_fees, cumulative_spread, cumulative_slippage
        if age[i] > config.max_quote_age_ms:
            order.update(status="expired", reason="quote_too_old")
            return
        delay_ms = (int(time_ns[i]) - order["_due"]) / 1_000_000
        if not order["is_liquidation"] and delay_ms > config.max_quote_delay_ms:
            order.update(status="expired", reason="quote_gap")
            return
        direction = 1 if order["_quantity"] > 0 else -1
        side_index = 1 if direction > 0 else 0
        quantity = min(order["requested_qty"], remaining[side_index])
        if quantity <= EPS:
            order.update(status="unfilled", reason="no_top_depth")
            return
        touch = ask[i] if direction > 0 else bid[i]
        price = touch * (1 + direction * config.slippage_bps / 10_000)
        fee = quantity * price * config.fee_bps / 10_000
        spread_cost = quantity * direction * (touch - mid[i])
        slippage_cost = quantity * direction * (price - touch)
        signed_quantity = direction * quantity
        episodes.record(times[i], signed_quantity, price, fee, position)
        cash -= signed_quantity * price + fee
        position += signed_quantity
        if abs(position) < EPS:
            position = 0.0
        remaining[side_index] -= quantity
        cumulative_fees += fee
        cumulative_spread += spread_cost
        cumulative_slippage += slippage_cost
        order["filled_qty"] = quantity
        full = quantity >= order["requested_qty"] - EPS
        order.update(status="filled" if full else "partial",
                     reason="filled" if full else "ioc_remainder_cancelled")
        fills.append({
            "timestamp": times[i], "decision_time": order["decision_time"],
            "order_id": order["order_id"], "side": order["side"],
            "quantity": quantity, "price": price, "mid_price": mid[i],
            "fee": fee, "spread_cost": spread_cost, "slippage_cost": slippage_cost,
            "latency_ms": (int(time_ns[i]) - int(order["decision_time"].value)) / 1_000_000,
            "is_liquidation": order["is_liquidation"],
        })

    for i in range(count):
        remaining = [bid_size[i], ask_size[i]]
        if pending is not None and pending["_arrival"] <= i:
            execute(pending, i, remaining)
            pending = None
        terminal = i == count - 1 and config.close_final_position
        if terminal:
            if pending is not None:
                pending.update(status="expired", reason="end_of_data")
                pending = None
            if abs(position) > EPS:
                execute(make_order(i, -position, liquidation=True), i, remaining)

        equity_value = cash + position * mid[i]
        net_pnl = equity_value - config.initial_cash
        history[i] = (
            position, cash, equity_value, net_pnl, net_pnl + cumulative_fees,
            cumulative_fees, cumulative_spread, cumulative_slippage,
        )
        if pending is None and not terminal:
            quantity = targets[i] - position
            if abs(quantity) > EPS:
                pending = make_order(i, quantity)

    if pending is not None:
        pending.update(status="expired", reason="end_of_data")
    equity = pd.DataFrame(history, columns=history_columns, copy=False)
    equity.insert(0, "mid_price", mid)
    equity.insert(0, "timestamp", times)
    fill_frame = pd.DataFrame(fills, columns=FILL_COLUMNS)
    trade_frame = pd.DataFrame(episodes.closed, columns=TRADE_COLUMNS)
    order_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    fill_frame = add_markouts(
        fill_frame, frame, horizons_ms=config.markout_horizons_ms,
        max_quote_age_ms=config.max_quote_age_ms,
    )
    metrics = compute_metrics(
        equity, trade_frame, fill_frame, order_frame,
        initial_cash=config.initial_cash, annualization_days=config.annualization_days,
    )
    metrics["clipped_target_count"] = int(np.sum(np.abs(np.asarray(
        _column(quotes, target_positions) if isinstance(target_positions, str) else target_positions,
        dtype=float,
    )) > config.max_abs_position))
    return BacktestResult(equity, fill_frame, trade_frame, order_frame, metrics, config)
