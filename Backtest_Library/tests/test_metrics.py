"""Analytical checks for account, episode, execution, and markout metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from Backtest_Library.metrics import add_markouts, compute_metrics


def curve(values, *, seconds=None, positions=None):
    seconds = range(len(values)) if seconds is None else seconds
    frame = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01", tz="UTC")
        + pd.to_timedelta(list(seconds), unit="s"),
        "equity": values,
    })
    if positions is not None:
        frame["position"] = positions
    return frame


def test_drawdown_includes_initial_capital_and_exposure_uses_elapsed_time():
    metrics = compute_metrics(
        curve([100, 110, 90], seconds=[0, 1, 11], positions=[0, 1, 0]),
        initial_cash=100,
    )
    assert metrics["net_pnl"] == -10
    assert metrics["total_return"] == pytest.approx(-0.1)
    assert metrics["max_drawdown"] == 20
    assert metrics["max_drawdown_pct"] == pytest.approx(20 / 110)
    assert metrics["time_in_market"] == pytest.approx(10 / 11)
    assert metrics["mean_abs_inventory"] == pytest.approx(10 / 11)
    assert metrics["final_position"] == 0

    initial_loss = compute_metrics(curve([95, 97]), initial_cash=100)
    assert initial_loss["max_drawdown"] == 5


def test_win_rate_uses_net_closed_episodes_including_breakeven():
    trades = pd.DataFrame({
        "net_pnl": [10.0, -5.0, 0.0], "holding_seconds": [1.0, 3.0, 8.0]
    })
    metrics = compute_metrics(curve([100, 110]), trades, initial_cash=100)
    assert metrics["closed_trades"] == 3
    assert metrics["win_rate"] == pytest.approx(1 / 3)
    assert metrics["profit_factor"] == 2
    assert metrics["expectancy"] == pytest.approx(5 / 3)
    assert metrics["breakeven_trades"] == 1
    assert metrics["closed_trade_net_pnl"] == 5
    assert metrics["open_trade_pnl"] == 5
    assert metrics["avg_holding_seconds"] == 4


def test_costs_are_reported_without_double_subtraction():
    equity = curve([100.0, 95.0])
    equity["cumulative_fees"] = [0.0, 1.0]
    equity["cumulative_spread_cost"] = [0.0, 3.0]
    equity["cumulative_slippage_cost"] = [0.0, 1.0]
    metrics = compute_metrics(equity, initial_cash=100)
    assert metrics["net_pnl"] == -5
    assert metrics["gross_pnl"] == -4
    assert metrics["spread_cost"] == 3
    assert metrics["slippage_cost"] == 1


def test_no_closed_trades_or_constant_daily_returns_have_undefined_ratios():
    equity = curve([100, 100, 100], seconds=[0, 86_400, 172_800])
    metrics = compute_metrics(equity, trades=pd.DataFrame(), initial_cash=100)
    assert metrics["daily_return_count"] == 3
    assert metrics["daily_sharpe"] is None
    assert metrics["daily_sortino"] is None
    assert metrics["win_rate"] is None
    assert metrics["profit_factor"] is None


def test_daily_risk_ratios_do_not_annualize_second_by_second_observations():
    equity = curve([110, 99, 108.9], seconds=[0, 86_400, 172_800])
    metrics = compute_metrics(equity, initial_cash=100)
    returns = np.array([0.1, -0.1, 0.1])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(365.25)
    expected_sortino = returns.mean() / math.sqrt((0.1 ** 2) / 3) * math.sqrt(365.25)
    assert metrics["daily_sharpe"] == pytest.approx(expected_sharpe)
    assert metrics["daily_sortino"] == pytest.approx(expected_sortino)

    one_day = compute_metrics(curve([100, 110, 105]), initial_cash=100)
    assert one_day["daily_return_count"] == 1
    assert one_day["daily_sharpe"] is None


def test_missing_days_are_not_mislabeled_as_daily_returns():
    metrics = compute_metrics(curve([110, 120], seconds=[0, 172_800]), initial_cash=100)
    assert metrics["daily_return_count"] == 1
    assert metrics["daily_sharpe"] is None


def test_effective_spread_preserves_sign_factor_two_and_quantity_weights():
    fills = pd.DataFrame({
        "quantity": [2.0, 1.0, 1.0], "price": [101.0, 99.0, 99.0],
        "mid_price": [100.0, 100.0, 100.0], "side": ["buy", "sell", "buy"],
    })
    orders = pd.DataFrame({"requested_qty": [2.0, 1.0, 1.0], "filled_qty": [2.0, 0.5, 0.0]})
    metrics = compute_metrics(curve([100, 100]), fills=fills, orders=orders, initial_cash=100)
    assert metrics["effective_spread_bps"] == pytest.approx(100)
    assert metrics["turnover_notional"] == 400
    assert metrics["fill_rate"] == pytest.approx(2 / 3)
    assert metrics["quantity_fill_ratio"] == pytest.approx(2.5 / 4)


def test_markouts_are_side_signed_and_use_only_quotes_at_or_before_horizon():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    quotes = pd.DataFrame({
        "timestamp": base + pd.to_timedelta([0, 1, 2, 3], unit="s"),
        "mid_price": [100.0, 102.0, 101.0, 104.0],
    })
    fills = pd.DataFrame({"timestamp": [base, base], "price": [101.0, 99.0], "side": ["buy", "sell"]})
    result = add_markouts(fills, quotes, horizons_ms=[1500, 5000])
    assert result["markout_bps_1500"].iloc[0] == pytest.approx((102 - 101) / 101 * 10_000)
    assert result["markout_bps_1500"].iloc[1] == pytest.approx(-(102 - 99) / 99 * 10_000)
    assert result["markout_bps_5000"].isna().all()
    assert "markout_bps_1500" not in fills.columns

    stale = add_markouts(fills, quotes, horizons_ms=[1500], max_quote_age_ms=400)
    assert stale["markout_bps_1500"].isna().all()


def test_markouts_handle_microsecond_timestamp_resolution():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    quotes = pd.DataFrame({
        "timestamp": pd.DatetimeIndex([base, base + pd.Timedelta(seconds=1)]).as_unit("us"),
        "mid_price": [100.0, 101.0],
    })
    fills = pd.DataFrame({"timestamp": [base], "price": [100.0], "side": ["buy"]})
    result = add_markouts(fills, quotes, horizons_ms=[1000])
    assert result["markout_bps_1000"].iloc[0] == pytest.approx(100)


def test_markouts_include_original_book_age_in_freshness_check():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    quotes = pd.DataFrame({
        "timestamp": base + pd.to_timedelta([0, 1, 2], unit="s"),
        "mid_price": [100.0, 101.0, 102.0],
        "book_age_ms": [0, 1900, 0],
    })
    fills = pd.DataFrame({"timestamp": [base], "price": [100.0], "side": ["buy"]})
    result = add_markouts(fills, quotes, horizons_ms=[1000, 1500], max_quote_age_ms=2000)
    assert result["markout_bps_1000"].iloc[0] == pytest.approx(100)
    assert result["markout_bps_1500"].isna().all()


def test_time_regressions_and_duplicate_quote_timestamps_are_rejected():
    with pytest.raises(ValueError, match="increasing and unique"):
        compute_metrics(curve([100, 101], seconds=[1, 0]), initial_cash=100)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    fills = pd.DataFrame({"timestamp": [base], "price": [100.0], "side": ["buy"]})
    quotes = pd.DataFrame({"timestamp": [base, base], "mid_price": [100.0, 101.0]})
    with pytest.raises(ValueError, match="increasing and unique"):
        add_markouts(fills, quotes)
