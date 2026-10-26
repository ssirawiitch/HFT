# LOB: Short-Horizon Price Prediction

This project uses Limit Order Book (LOB) and trade data to predict whether the
BTCUSDT perpetual mid-price will rise, fall, or stay roughly unchanged over the
next second.

We will compare Logistic Regression and LightGBM using features such as bid–ask
imbalance, spread, and recent trade flow. Chronological backtests will evaluate
whether predictions remain useful after trading fees and latency, with position
limits to control risk.

## Data

Free historical data from Bybit:

- [BTCUSDT order book](https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/)
- [BTCUSDT trades](https://public.bybit.com/trading/BTCUSDT/)

Start with one matching day of both datasets, then expand to 30–90 days. Check
each file's schema before processing. Data files are not included in this folder.

## Reference

[Queue Imbalance as a One-Tick-Ahead Price Predictor](https://arxiv.org/abs/1512.03492)
provides the starting idea: use bid–ask queue imbalance to predict price movement.
