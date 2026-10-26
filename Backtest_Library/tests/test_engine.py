"""Independent, hand-calculated checks for quote replay and trade accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Backtest_Library import BacktestConfig, run_backtest


def quote_frame(
    n=3,
    *,
    seconds=None,
    bids=None,
    asks=None,
    bid_qty=10.0,
    ask_qty=10.0,
):
    if seconds is None:
        seconds = np.arange(n)
    n = len(seconds)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC")
            + pd.to_timedelta(seconds, unit="s"),
            "best_bid": np.full(n, 99.0) if bids is None else bids,
            "bid_qty": np.full(n, bid_qty),
            "best_ask": np.full(n, 101.0) if asks is None else asks,
            "ask_qty": np.full(n, ask_qty),
        }
    )


def config(**overrides):
    values = {
        "initial_cash": 1000.0,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "latency_ms": 0,
        "max_abs_position": 10.0,
        "max_quote_fraction": 1.0,
        "max_quote_age_ms": 2000,
        "max_quote_delay_ms": 2000,
        "close_final_position": False,
        "markout_horizons_ms": (),
    }
    values.update(overrides)
    return BacktestConfig(**values)


def final_value(result, name):
    return float(result.equity[name].iloc[-1])


def assert_cashflow_identity(result, initial_cash=1000.0):
    """Reconstruct cash/inventory directly from signed executed cash flows."""
    equity = result.equity
    timestamps = pd.DatetimeIndex(equity["timestamp"])
    if result.fills.empty:
        expected_position = np.zeros(len(equity))
        expected_cash = np.full(len(equity), initial_cash)
    else:
        fills = result.fills.copy()
        fills["signed_quantity"] = fills["quantity"] * fills["side"].map(
            {"buy": 1.0, "sell": -1.0}
        )
        fills["cash_change"] = (
            -fills["signed_quantity"] * fills["price"] - fills["fee"]
        )
        by_time = fills.groupby("timestamp")[["signed_quantity", "cash_change"]].sum()
        by_time = by_time.reindex(timestamps, fill_value=0.0)
        expected_position = by_time["signed_quantity"].cumsum().to_numpy()
        expected_cash = initial_cash + by_time["cash_change"].cumsum().to_numpy()
    np.testing.assert_allclose(equity["position"], expected_position, atol=1e-10)
    np.testing.assert_allclose(equity["cash"], expected_cash, atol=1e-10)
    np.testing.assert_allclose(
        equity["equity"],
        equity["cash"] + equity["position"] * equity["mid_price"],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        equity["net_pnl"], equity["equity"] - initial_cash, atol=1e-10
    )
    np.testing.assert_allclose(
        equity["gross_pnl"],
        equity["net_pnl"] + equity["cumulative_fees"],
        atol=1e-10,
    )


def test_no_trading_keeps_initial_equity_and_undefined_win_rate():
    result = run_backtest(quote_frame(n=4), [0, 0, 0, 0], config=config())
    assert result.fills.empty
    assert result.trades.empty
    assert result.orders.empty
    assert result.metrics["closed_trades"] == 0
    assert result.metrics["win_rate"] is None
    assert final_value(result, "net_pnl") == pytest.approx(0.0)
    assert_cashflow_identity(result)


def test_constant_market_round_trip_loses_spread_and_both_fees():
    result = run_backtest(quote_frame(), [1, 0, 0], config=config(fee_bps=10))
    assert result.fills["side"].tolist() == ["buy", "sell"]
    np.testing.assert_allclose(result.fills["price"], [101.0, 99.0])
    np.testing.assert_allclose(result.fills["fee"], [0.101, 0.099])
    assert final_value(result, "equity") == pytest.approx(997.8)
    assert final_value(result, "gross_pnl") == pytest.approx(-2.0)
    assert final_value(result, "net_pnl") == pytest.approx(-2.2)
    assert final_value(result, "cumulative_fees") == pytest.approx(0.2)
    assert result.fills["spread_cost"].sum() == pytest.approx(2.0)
    assert len(result.trades) == 1
    assert result.trades["net_pnl"].iloc[0] == pytest.approx(-2.2)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert_cashflow_identity(result)


def test_adverse_slippage_is_in_fill_prices_without_double_subtraction():
    result = run_backtest(quote_frame(), [1, 0, 0], config=config(slippage_bps=100))
    np.testing.assert_allclose(result.fills["price"], [102.01, 98.01])
    assert result.fills["spread_cost"].sum() == pytest.approx(2.0)
    assert result.fills["slippage_cost"].sum() == pytest.approx(2.0)
    assert final_value(result, "net_pnl") == pytest.approx(-4.0)
    assert final_value(result, "gross_pnl") == pytest.approx(-4.0)
    assert_cashflow_identity(result)


def test_zero_configured_latency_still_uses_strictly_next_quote():
    quotes = quote_frame(bids=[99.0, 109.0, 109.0], asks=[101.0, 111.0, 111.0])
    result = run_backtest(quotes, [1, 0, 0], config=config(latency_ms=0))
    np.testing.assert_allclose(result.fills["price"], [111.0, 109.0])
    assert result.fills["timestamp"].tolist() == quotes["timestamp_utc"].iloc[1:].tolist()
    assert (result.fills["timestamp"] > result.fills["decision_time"]).all()
    # A same-row fill would incorrectly turn this into an eight-unit profit.
    assert final_value(result, "net_pnl") == pytest.approx(-2.0)
    assert_cashflow_identity(result)


def test_reversal_splits_one_fill_between_two_closed_episodes_and_allocates_fees():
    result = run_backtest(quote_frame(n=4), [1, -1, 0, 0], config=config(fee_bps=10))
    assert result.fills["side"].tolist() == ["buy", "sell", "buy"]
    np.testing.assert_allclose(result.fills["quantity"], [1.0, 2.0, 1.0])
    assert result.trades["side"].tolist() == ["long", "short"]
    np.testing.assert_allclose(result.trades["quantity"], [1.0, 1.0])
    np.testing.assert_allclose(result.trades["gross_pnl"], [-2.0, -2.0])
    np.testing.assert_allclose(result.trades["fees"], [0.2, 0.2])
    np.testing.assert_allclose(result.trades["net_pnl"], [-2.2, -2.2])
    assert result.metrics["closed_trades"] == 2
    assert final_value(result, "net_pnl") == pytest.approx(-4.4)
    assert result.trades["net_pnl"].sum() == pytest.approx(final_value(result, "net_pnl"))
    assert_cashflow_identity(result)


def test_scale_in_and_partial_exits_form_one_position_episode():
    quotes = quote_frame(
        n=5,
        bids=[98.0, 98.0, 108.0, 120.0, 130.0],
        asks=[99.0, 100.0, 110.0, 122.0, 132.0],
    )
    result = run_backtest(quotes, [1, 2, 1, 0, 0], config=config(fee_bps=100))
    np.testing.assert_allclose(result.fills["price"], [100.0, 110.0, 120.0, 130.0])
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["quantity"] == pytest.approx(2.0)
    assert trade["entry_price"] == pytest.approx(105.0)
    assert trade["exit_price"] == pytest.approx(125.0)
    assert trade["gross_pnl"] == pytest.approx(40.0)
    assert trade["fees"] == pytest.approx(4.6)
    assert trade["net_pnl"] == pytest.approx(35.4)
    assert final_value(result, "net_pnl") == pytest.approx(35.4)
    assert_cashflow_identity(result)


def test_partial_ioc_orders_retry_without_creating_extra_closed_trades():
    quotes = quote_frame(n=8, bid_qty=0.3, ask_qty=0.4)
    result = run_backtest(quotes, [1, 1, 1, 0, 0, 0, 0, 0], config=config())
    np.testing.assert_allclose(
        result.fills["quantity"], [0.4, 0.4, 0.2, 0.3, 0.3, 0.3, 0.1], atol=1e-10
    )
    assert len(result.orders) == 7
    assert (result.orders["status"] == "partial").sum() == 5
    assert result.orders["requested_qty"].sum() == pytest.approx(4.0)
    assert result.orders["filled_qty"].sum() == pytest.approx(2.0)
    assert len(result.trades) == 1
    assert final_value(result, "position") == pytest.approx(0.0, abs=1e-10)
    assert final_value(result, "net_pnl") == pytest.approx(-2.0)
    assert_cashflow_identity(result)


def test_open_position_is_marked_to_mid_but_not_counted_as_closed_trade():
    result = run_backtest(quote_frame(n=2), [1, 1], config=config(fee_bps=10))
    assert len(result.fills) == 1
    assert result.trades.empty
    assert result.metrics["closed_trades"] == 0
    assert result.metrics["win_rate"] is None
    assert final_value(result, "position") == pytest.approx(1.0)
    assert final_value(result, "net_pnl") == pytest.approx(-1.101)
    assert_cashflow_identity(result)


def test_optional_terminal_liquidation_is_explicit_and_closes_at_bid():
    result = run_backtest(
        quote_frame(n=2), [1, 1], config=config(close_final_position=True)
    )
    assert len(result.fills) == 2
    assert not bool(result.fills["is_liquidation"].iloc[0])
    assert bool(result.fills["is_liquidation"].iloc[1])
    assert result.fills["timestamp"].iloc[0] == result.fills["timestamp"].iloc[1]
    np.testing.assert_allclose(result.fills["price"], [101.0, 99.0])
    assert final_value(result, "position") == pytest.approx(0.0)
    assert final_value(result, "net_pnl") == pytest.approx(-2.0)
    assert len(result.trades) == 1
    assert_cashflow_identity(result)


def test_terminal_liquidation_cannot_consume_same_side_depth_twice():
    result = run_backtest(
        quote_frame(bid_qty=0.3, ask_qty=1.0),
        [1, 0, 0],
        config=config(close_final_position=True),
    )
    # The regular sell at t=2 consumes all 0.3 displayed bid depth. The terminal
    # liquidation must leave the remaining 0.7 position rather than fill again.
    sells = result.fills.loc[result.fills["side"] == "sell"]
    assert sells["quantity"].sum() == pytest.approx(0.3)
    assert final_value(result, "position") == pytest.approx(0.7)
    assert final_value(result, "net_pnl") == pytest.approx(-1.3)
    assert result.trades.empty
    assert_cashflow_identity(result)


def test_positive_latency_waits_until_first_available_quote_after_arrival():
    quotes = quote_frame(
        n=5,
        bids=[99.0, 109.0, 119.0, 129.0, 139.0],
        asks=[101.0, 111.0, 121.0, 131.0, 141.0],
    )
    result = run_backtest(quotes, [1, 1, 0, 0, 0], config=config(latency_ms=1500))
    np.testing.assert_allclose(result.fills["price"], [121.0, 139.0])
    assert result.fills["timestamp"].tolist() == quotes["timestamp_utc"].iloc[[2, 4]].tolist()
    np.testing.assert_allclose(result.fills["latency_ms"], [2000.0, 2000.0])
    assert final_value(result, "net_pnl") == pytest.approx(18.0)
    assert_cashflow_identity(result)


def test_order_expires_instead_of_filling_across_a_large_quote_gap():
    quotes = quote_frame(seconds=[0, 10, 11])
    result = run_backtest(quotes, [1, 0, 0], config=config(max_quote_delay_ms=2000))
    assert result.fills.empty
    assert len(result.orders) == 1
    assert result.orders["status"].iloc[0] == "expired"
    assert final_value(result, "net_pnl") == pytest.approx(0.0)
    assert_cashflow_identity(result)


def test_last_row_signal_cannot_execute_without_a_future_quote():
    result = run_backtest(quote_frame(n=2), [0, 1], config=config())
    assert result.fills.empty
    assert result.trades.empty
    assert final_value(result, "position") == pytest.approx(0.0)
    assert final_value(result, "net_pnl") == pytest.approx(0.0)
    if not result.orders.empty:
        assert result.orders["status"].isin(["expired", "unfilled"]).all()


def test_requested_position_is_clipped_to_risk_limit():
    result = run_backtest(quote_frame(), [99, 0, 0], config=config(max_abs_position=0.25))
    assert result.equity["position"].abs().max() <= 0.25 + 1e-12
    np.testing.assert_allclose(result.fills["quantity"], [0.25, 0.25])
    assert final_value(result, "net_pnl") == pytest.approx(-0.5)
    assert_cashflow_identity(result)


def test_zero_position_limit_disables_trading():
    result = run_backtest(quote_frame(), [1, -1, 1], config=config(max_abs_position=0))
    assert result.fills.empty
    assert result.orders.empty
    assert final_value(result, "net_pnl") == pytest.approx(0.0)
    assert_cashflow_identity(result)


def test_fill_size_respects_configured_fraction_of_displayed_top_liquidity():
    result = run_backtest(
        quote_frame(), [2, 2, 2], config=config(max_quote_fraction=0.1)
    )
    np.testing.assert_allclose(result.fills["quantity"], [1.0, 1.0])
    assert final_value(result, "position") == pytest.approx(2.0)
    assert_cashflow_identity(result)


def test_stale_quotes_are_not_used_for_execution():
    quotes = quote_frame(n=4)
    quotes["book_age_ms"] = [0, 5000, 0, 0]
    result = run_backtest(quotes, [1, 1, 0, 0], config=config(max_quote_age_ms=2000))
    assert not (result.fills["timestamp"] == quotes["timestamp_utc"].iloc[1]).any()
    assert_cashflow_identity(result)


def test_target_column_and_materialized_polars_quotes_are_supported():
    pl = pytest.importorskip("polars")
    quotes = quote_frame()
    quotes["target_position"] = [1.0, 0.0, 0.0]
    polars_quotes = pl.DataFrame({name: quotes[name].tolist() for name in quotes.columns})
    result = run_backtest(polars_quotes, config=config())
    assert len(result.fills) == 2
    assert final_value(result, "net_pnl") == pytest.approx(-2.0)
    assert_cashflow_identity(result)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_time",
        "unsorted_time",
        "missing_time",
        "crossed_book",
        "non_finite_price",
        "negative_quantity",
    ],
)
def test_invalid_quote_inputs_are_rejected(change):
    quotes = quote_frame()
    if change == "duplicate_time":
        quotes.loc[1, "timestamp_utc"] = quotes.loc[0, "timestamp_utc"]
    elif change == "unsorted_time":
        quotes = quotes.iloc[::-1].reset_index(drop=True)
    elif change == "missing_time":
        quotes.loc[1, "timestamp_utc"] = pd.NaT
    elif change == "crossed_book":
        quotes.loc[1, "best_bid"] = 102.0
    elif change == "non_finite_price":
        quotes.loc[1, "best_ask"] = np.nan
    elif change == "negative_quantity":
        quotes.loc[1, "bid_qty"] = -1.0
    with pytest.raises(ValueError):
        run_backtest(quotes, [1, 0, 0], config=config())


@pytest.mark.parametrize("targets", [[1, 0], [1, np.nan, 0], [1, np.inf, 0]])
def test_invalid_targets_are_rejected(targets):
    with pytest.raises(ValueError):
        run_backtest(quote_frame(), targets, config=config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_cash": 0},
        {"fee_bps": -1},
        {"slippage_bps": -1},
        {"latency_ms": -1},
        {"max_abs_position": -1},
        {"max_quote_fraction": 1.1},
        {"max_quote_age_ms": -1},
        {"max_quote_delay_ms": -1},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        run_backtest(quote_frame(), [1, 0, 0], config=config(**overrides))
