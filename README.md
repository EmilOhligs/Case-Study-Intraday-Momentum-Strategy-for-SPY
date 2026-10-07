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
- **Costs:** $0.0035/share commission (IBKR, at least $0.35 per order) plus $0.001/share slippage, as in the paper. Plus a cost-sensitivity curve up to $0.0235/share.
- **Metrics:** annualized return (CAGR), annualized volatility (σ·√252), Sharpe ratio (r_f = 0), max drawdown, hit ratio, skewness, alpha/beta vs SPY.

## Documentation

- [`docs/strategy.md`](docs/strategy.md): mathematical specification. Part I is the paper's strategy and evaluation, Part II is our ML model (features, estimation, CV, AUC).
- [`docs/extensions.md`](docs/extensions.md): own strategy (ML long/short/flat model): design, hypotheses, validation, results
- [`docs/report.md`](docs/report.md): case study report covering data, implementation decisions, **cross-check with the authors' reference code**, validation, results and limitations

## Repository structure

```
src/intraday_momentum/
    config.py      strategy & cost parameters (dataclasses)
    data.py        load minute bars -> day x minute matrices (time convention documented here)
    signals.py     Noise Area, VWAP-based stops, trailing daily volatility
    backtest.py    daily event loop: decisions every 30 min, flat at close, per-share costs
    metrics.py     Sharpe, CAGR, volatility, drawdown, alpha/beta
    evaluation.py  train/test evaluation, cost sensitivity, parameter grid (shared by notebook and script)
    diagnostics.py trade-level analysis: false breakouts, P&L by entry time / side / year
    features.py    ML features (no look-ahead) and labels
    ml_strategy.py own strategy: logistic regression, expanding-window CV, evaluation
    plotting.py    figures
    synthetic.py   random-walk minute bars for tests
notebooks/
    backtest.ipynb     main analysis notebook (imports the package, no duplicated logic)
scripts/
    download_data.py   Alpaca download
    run_backtest.py    paper replication -> results/
    run_ml.py          own ML strategy -> results/ml_*
tests/                 pytest suite (look-ahead, P&L accounting, metrics)
results/               tables and figures
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env               # add your Alpaca API keys
python scripts/download_data.py    # ~10 years of SPY minute bars into data/
jupyter lab notebooks/backtest.ipynb   # interactive analysis
python scripts/run_backtest.py     # paper replication, headless -> results/
python scripts/run_ml.py           # own ML strategy, headless -> results/ml_*
pytest                             # run the tests
```

`python scripts/run_backtest.py --synthetic` runs the whole pipeline on synthetic data (no API key needed).

## Results (SPY 1-min, train 2016–2021, test 2022-01 – 2026-10, after costs)

| Strategy | Sharpe train | Sharpe test | Ann. return test | Max DD test |
|---|---|---|---|---|
| Base: opposite-band stop | 0.37 | 0.74 | 6.8% | 8.9% |
| Ext. 1: + band/VWAP stop | 0.66 | 1.02 | 6.7% | 10.0% |
| Ext. 2: + vol targeting (full paper model) | 0.93 | 1.10 | 15.8% | 22.3% |
| **Own: ML logistic (1x)** | 0.02 (in-sample) | **0.59** | 5.7% | 12.8% |
| **Own: ML logistic + vol targeting** | 0.29 (in-sample) | **0.71** | 13.3% | 20.5% |
| SPY buy & hold | 1.05 | 0.75 | 12.2% | 24.5% |

- **Replication:** yearly returns match the paper's table with a correlation of 0.99. The ranking of the variants holds out of sample, and the edge survives considerably higher costs (sensitivity curve).
- **After publication** (2024-05 on) the paper rule's Sharpe is about 0, while the ML model keeps 0.51.
- **ML vs. rule:** the ML model has a small out-of-sample edge (AUC 0.518) but does not beat the rule over the full test period. Its returns are uncorrelated with the rule (ρ = −0.03), so combining the two is the most promising next step.

Details: [`docs/report.md`](docs/report.md), [`docs/extensions.md`](docs/extensions.md), `results/`.
