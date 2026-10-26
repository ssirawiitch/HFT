# Backtest Library

Our own Python functions for sampled-order-book backtests, performance metrics,
and Matplotlib reports. The core engine is implemented here; it does not use an
external backtesting framework. Python 3.10+ is supported.

## Install and import

From the HFT project root:

```bash
python -m pip install -e ./Backtest_Library
```

In a notebook opened from `LOB/`, use `%pip install -e ../Backtest_Library`.
The example notebook also finds the project root automatically.
In Jupyter, run `%matplotlib inline` once before displaying charts.

```python
import numpy as np
from Backtest_Library import BacktestConfig, run_backtest, plot_dashboard

# df may be our Polars table or a pandas DataFrame.
imbalance = (
    (df["bid_qty"].to_numpy() - df["ask_qty"].to_numpy())
    / (df["bid_qty"].to_numpy() + df["ask_qty"].to_numpy())
)
target = np.where(imbalance > 0.30, 0.001,
                  np.where(imbalance < -0.30, -0.001, 0.0))

result = run_backtest(df, target_positions=target, config=BacktestConfig(
    initial_cash=10_000,
    fee_bps=5.0,             # Example assumption; set your venue/tier's fees.
    slippage_bps=0.1,
    latency_ms=100,
    max_abs_position=0.001,  # BTC, not a portfolio weight.
    max_quote_fraction=0.10,
))
display(result.summary())
fig = plot_dashboard(result, save_path="backtest_dashboard.png")
display(fig)
```

The imbalance rule is a demonstration of the API, not an established profitable
strategy. Start with a short data slice when exploring new rules.

## Inputs and execution

Required quote columns are `timestamp_utc` (or `timestamp`), `best_bid`,
`bid_qty`, `best_ask`, and `ask_qty`. Timestamps must be sorted and unique;
`book_age_ms` is optional. Naive datetimes are interpreted as UTC. Target
positions are one base-asset quantity per row, or the name of a column containing
those quantities. Inputs are not modified or silently reordered.

- Targets are clipped to `max_abs_position`.
- One IOC taker order may be in flight. New target decisions wait for it to
  resolve. A partial fill cancels its remainder; later decisions can retry.
- Buy at a future ask and sell at a future bid, with adverse slippage and fees.
  Displayed top-level quantities cap fills; `max_quote_fraction` limits how much
  of that displayed quantity may be taken per observation.
- Orders always use a **strictly later quote**, including when `latency_ms=0`.
  With one-second data, 100 ms latency still executes no sooner than the next
  available second. `fills.latency_ms` reports the actual simulated delay.
- Quotes too old or arriving too far after the desired execution time expire
  orders. The default age/delay tolerances are 2,000 ms.
- Open positions are marked to the midpoint at the end. Optional
  `close_final_position=True` performs a flagged same-final-quote liquidation;
  this explicitly bypasses ordinary latency, shares the same displayed-depth
  budget, and may leave residual inventory.

This is a small-order **sampled-quote taker model**. It does not model maker
queues, market impact, funding, margin/liquidation rules, or hidden liquidity.
One-second quotes do not establish microsecond HFT execution performance.

## Results and metric definitions

`result.equity`, `result.fills`, `result.trades`, and `result.orders` are pandas
DataFrames. `result.metrics` is a dictionary and `result.summary()` is a Series.
`result.save("report_folder")` exports all ledgers plus a JSON summary.

| Group | Measurements |
| --- | --- |
| PnL and risk | Gross/net PnL, return, fees, max drawdown, daily Sharpe/Sortino |
| Closed positions | Win rate, profit factor, expectancy, average win/loss, holding time |
| Activity | Fill/order counts, turnover, net edge in bps, PnL per million traded |
| Execution | Filled-order rate, quantity fill ratio, actual latency, effective spread |
| Inventory | Maximum/mean/RMS inventory, time-weighted exposure, final position |
| Post-fill behavior | Quantity-weighted signed markouts at 1, 5, and 10 seconds |

Win rate counts **closed flat-to-flat position episodes after fees**, including
breakeven episodes in its denominator. Scale-ins and partial exits belong to the
same episode. A reversal closes one episode and starts another, with its fee
allocated between them. Open episodes are excluded from win rate.

Net PnL is `cash + position * mid - initial_cash`. Gross PnL adds back fees only:
spread and slippage already affect execution prices and must not be deducted a
second time. `open_trade_pnl` means account PnL attributable to unfinished
episodes; it can include partial-exit realized gains and is not pure unrealized
PnL.

Daily Sharpe/Sortino use observed UTC-day end returns and zero risk-free/target
return, conventionally annualized with 365.25 days for crypto. Partial first/last
days are included; missing days are not silently treated as one-day returns.
Insufficient observations or undefined ratios return `None`. This estimate does
not correct for autocorrelation. Percentage metrics are fractions: `0.10 = 10%`.

Effective spread is quantity weighted and uses
`2 * side * (fill_price - fill_time_mid) / fill_time_mid * 10_000`.
This fill-time benchmark differs from regulatory receipt-time definitions.
Markout is `side * (future_mid - fill_price) / fill_price * 10_000`;
positive values are favorable. Future midpoints use only quotes available at or
before the requested horizon, with quote-age checks; unavailable horizons are
missing. Inventory exposure weights each interval by its actual duration.

## Charts and example

All plot functions return a Figure and accept `save_path=...` and `show=False`:

- `plot_dashboard`: six panels covering PnL, drawdown, inventory, closed-position
  outcomes, trade PnL distribution, and markouts or execution costs.
- `plot_pnl`, `plot_drawdown`, `plot_inventory`, `plot_trade_distribution`,
  `plot_win_rate`.
- `plot_execution`: costs, latency, order status, and effective spread.
- `plot_markouts`: post-fill price behavior by horizon.

Plotting reduces very large time series while retaining bucket extremes;
metrics always use the full result. Open `examples/01_backtest_demo.ipynb` for a
worked example using our LOB Parquet file. Saved example figures and CSV/JSON
reports are in `examples/output/`.

Run validation from the project root:

```bash
python -m pytest Backtest_Library/tests
```

References for the assumptions and definitions:
[replay fills and queue limitations](https://hftbacktest.readthedocs.io/en/latest/order_fill.html),
[Sharpe ratio and time aggregation](https://www-leland.stanford.edu/~wfsharpe/art/sr/SR.htm),
[SEC execution quality definitions](https://www.sec.gov/reports/report-practice-preferencing).
