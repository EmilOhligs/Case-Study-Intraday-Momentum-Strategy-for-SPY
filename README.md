# Intraday Momentum Strategy for SPY

Replication and extension of **Zarattini, Aziz & Barbon (2024), *Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)*** – WUTIS Algorithmic Trading case study.

## Strategy in one paragraph

Every day, a time-of-day dependent **Noise Area** is placed around the open: its width at minute *HH:MM* is the average absolute move from the open to *HH:MM* over the previous 14 days (gap-adjusted with yesterday's close). A move outside this area signals a demand/supply imbalance: **long above the upper band, short below the lower band**. Decisions are taken only at HH:00 and HH:30, and every position is closed at 16:00 (no overnight risk).

| Variant | Exit / stop | Sizing |
|---|---|---|
| **Base** | hold until the opposite band is crossed | 100% of equity |
| **Extension 1** | trailing stop at max(UB, VWAP) for longs / min(LB, VWAP) for shorts | 100% of equity |
| **Extension 2** (= full paper model) | as Extension 1 | volatility targeting: 2% daily vol, leverage ≤ 4x |

## Methodology

- **Data:** SPY 1-minute bars (consolidated SIP feed, unadjusted) from Alpaca, 2016 onwards; regular session only. Benchmark: SPY buy & hold from split- and dividend-adjusted daily bars.
- **Train/test split:** train 2016–2021, test 2022–today. All variants use the **paper's parameters**, so nothing is fitted on the test set. A lookback × volatility-multiplier grid is evaluated **on the train period only** as a robustness check.
- **No look-ahead:** the signal at 10:00 uses the close of the 09:59 bar; the trade is executed at the **open of the next bar**. The Noise Area and the volatility estimate use only previous days. Both properties are unit-tested (`tests/test_signals.py`).
- **Costs:** $0.0035/share commission (IBKR) plus slippage. *Paper* scenario: $0.001/share. *Conservative* scenario: $0.005/share (≈ half the SPY bid-ask spread). Plus a full cost-sensitivity curve.
- **Metrics:** annualized return (CAGR), annualized volatility (σ·√252), Sharpe ratio (r_f = 0), max drawdown, hit ratio, skewness, alpha/beta vs SPY.

## Repository structure

```
src/intraday_momentum/
    config.py      strategy & cost parameters (dataclasses)
    data.py        load minute bars -> day x minute matrices (time convention documented here)
    signals.py     Noise Area, VWAP-based stops, trailing daily volatility
    backtest.py    daily event loop: decisions every 30 min, flat at close, per-share costs
    metrics.py     Sharpe, CAGR, volatility, drawdown, alpha/beta
    plotting.py    figures
    synthetic.py   random-walk minute bars for tests
scripts/
    download_data.py   Alpaca download
    run_backtest.py    full evaluation -> results/
tests/                 pytest suite (look-ahead, P&L accounting, metrics)
results/               tables and figures
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env               # add your Alpaca API keys
python scripts/download_data.py    # ~10 years of SPY minute bars into data/
python scripts/run_backtest.py     # writes results/
pytest                             # run the tests
```

`python scripts/run_backtest.py --synthetic` runs the whole pipeline on synthetic data (no API key needed).

## Results

*Filled in after running on real data – see `results/summary.md`.*
