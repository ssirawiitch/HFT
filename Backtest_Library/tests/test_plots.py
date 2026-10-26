"""Plot API checks: figures, empty results and financial chart conventions."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from Backtest_Library import plots


PLOT_FUNCTIONS = [
    plots.plot_dashboard, plots.plot_pnl, plots.plot_drawdown,
    plots.plot_trade_distribution, plots.plot_inventory,
    plots.plot_execution, plots.plot_markouts, plots.plot_win_rate,
]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def sample_result(empty=False):
    if empty:
        return SimpleNamespace(equity=pd.DataFrame(), fills=pd.DataFrame(),
                               trades=pd.DataFrame(), orders=pd.DataFrame(),
                               metrics={}, config=SimpleNamespace(initial_cash=1000.0))
    timestamps = pd.date_range("2026-09-24", periods=100, freq="s", tz="UTC")
    net = np.sin(np.arange(100) / 10) + np.arange(100) / 20
    fees = np.linspace(0, 1, 100)
    equity = pd.DataFrame({
        "timestamp": timestamps, "equity": 1000 + net, "net_pnl": net,
        "gross_pnl": net + fees, "cumulative_fees": fees,
        "position": np.sin(np.arange(100) / 12) * 0.01,
    })
    fills = pd.DataFrame({
        "timestamp": timestamps[[10, 40]], "side": ["buy", "sell"],
        "quantity": [1.0, 3.0], "price": [100.1, 99.9], "mid_price": [100.0, 100.0],
        "fee": [0.01, 0.03], "spread_cost": [0.1, 0.3],
        "slippage_cost": [0.0, 0.0], "latency_ms": [1000.0, 1100.0],
        "markout_bps_1000": [1.0, 5.0], "markout_bps_5000": [-2.0, 2.0],
    })
    return SimpleNamespace(
        equity=equity, fills=fills, trades=pd.DataFrame({"net_pnl": [3.0, -1.0, 0.0]}),
        orders=pd.DataFrame({"status": ["filled", "partial", "unfilled", "expired"]}),
        metrics={"net_pnl": float(net[-1]), "total_return": float(net[-1] / 1000), "daily_sharpe": None},
        config=SimpleNamespace(initial_cash=1000.0),
    )


@pytest.mark.parametrize("plotter", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_public_plotters_save_figures(tmp_path, plotter, empty):
    output = tmp_path / f"{plotter.__name__}_{empty}.png"
    figure = plotter(sample_result(empty), save_path=output)
    assert isinstance(figure, Figure)
    assert figure.axes
    assert output.is_file() and output.stat().st_size > 0


def test_markouts_are_quantity_weighted():
    figure = plots.plot_markouts(sample_result())
    values = [bar.get_height() for bar in figure.axes[0].patches]
    assert values == pytest.approx([4.0, 1.0])


def test_outcomes_match_metric_roundoff_tolerance():
    result = sample_result()
    result.trades = pd.DataFrame({"net_pnl": [1.0, -1.0, 0.0, 5e-13, -5e-13]})
    figure = plots.plot_trade_distribution(result)
    assert [bar.get_height() for bar in figure.axes[1].patches] == [1, 1, 3]
    assert "20.0%" in figure.axes[1].get_title(loc="left")


def test_drawdown_includes_initial_capital():
    result = sample_result()
    result.equity = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-24", periods=2, freq="s", tz="UTC"),
        "equity": [990.0, 980.0], "net_pnl": [-10.0, -20.0],
    })
    figure = plots.plot_drawdown(result)
    assert figure.axes[0].lines[0].get_ydata() == pytest.approx([-1.0, -2.0])


def test_plot_decimation_retains_local_extremes():
    x = np.arange(100_001)
    pnl = np.zeros(len(x))
    pnl[12_345], pnl[54_321] = -1000.0, 2000.0
    sampled_x, sampled = plots._decimate(x, {"pnl": pnl, "fees": x.astype(float)}, 1000)
    assert len(sampled_x) <= 1000
    assert sampled_x[0] == 0 and sampled_x[-1] == x[-1]
    assert 12_345 in sampled_x and 54_321 in sampled_x
    assert sampled["pnl"].min() == -1000.0
    assert sampled["pnl"].max() == 2000.0


def test_plotting_does_not_change_global_style():
    before = dict(matplotlib.rcParams)
    plots.plot_dashboard(sample_result())
    assert dict(matplotlib.rcParams) == before


def test_near_constant_execution_cost_histogram_remains_visible():
    result = sample_result()
    spreads = np.array([0.211800, 0.211801, 0.211802, 0.211803])
    mid = np.full(len(spreads), 100.0)
    result.fills = pd.DataFrame({
        "side": ["buy"] * len(spreads), "quantity": [1.0, 2.0, 3.0, 4.0],
        "price": mid * (1 + spreads / 20_000), "mid_price": mid,
        "latency_ms": [1000.0] * len(spreads),
    })
    figure = plots.plot_execution(result)
    latency_ax, spread_ax = figure.axes[1], figure.axes[3]
    lower, upper = spread_ax.get_xlim()
    assert lower > 0.2  # Zero must not drag the histogram window away from its data.
    assert lower < spreads.min() and upper > spreads.max()
    assert upper - lower < 0.001
    assert all(patch.get_width() > 0 for patch in spread_ax.patches)
    assert any("Qty-weighted mean" in text.get_text() for text in spread_ax.texts)
    figure.canvas.draw()
    assert not latency_ax.xaxis.get_major_formatter().get_useOffset()
    assert not spread_ax.xaxis.get_major_formatter().get_useOffset()
    assert latency_ax.xaxis.get_offset_text().get_text() == ""
    assert spread_ax.xaxis.get_offset_text().get_text() == ""
