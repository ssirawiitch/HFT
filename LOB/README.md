# LOB: Short-Horizon Price Prediction

This project uses Limit Order Book (LOB) data to predict whether the
BTCUSDT perpetual mid-price will rise, fall, or stay roughly unchanged over the
next second.

We will compare Logistic Regression and LightGBM using features such as bid–ask
imbalance, spread, and recent price changes. Chronological backtests will evaluate
whether predictions remain useful after trading fees and latency, with position
limits to control risk.

## Data

Free historical data from Bybit:

- [BTCUSDT order book](https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/)

Raw `.data` files are stored in `../Data/`. Trade data is optional for this first
phase.

Open [main.ipynb](01_data_preview.ipynb) and run the cells in order.
It previews the raw files, then reconstructs every day from snapshots and deltas.
The combined `df` contains best bid, best ask, quantities, mid-price, spread, and
book age at one-second UTC boundaries, sorted by `timestamp_utc`.

The complete result is saved to `processed/btcusdt_lob_1s.parquet`. Daily caches
avoid reprocessing unchanged raw files. The last notebook cell reloads the saved
data into `df` for further analysis. This loader runs independently and finds the
saved file from the project root, `LOB/`, or a project subfolder.

## Reference

[Queue Imbalance as a One-Tick-Ahead Price Predictor](https://arxiv.org/abs/1512.03492)
provides the starting idea: use bid–ask queue imbalance to predict price movement.
