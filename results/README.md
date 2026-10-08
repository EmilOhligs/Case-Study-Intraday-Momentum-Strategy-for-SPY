# Results

All files here are written by the scripts and the notebook. SPY 1-minute data, train 2016–2021, test 2022-01 – 2026-10, after costs.

## Tables

| File | Content | Written by |
|---|---|---|
| [`summary.md`](summary.md) | Paper replication: Base, Ext. 1, Ext. 2 and SPY buy & hold, train and test | `scripts/run_backtest.py` |
| [`ml_summary.md`](ml_summary.md) | Own ML strategy vs. the paper rule, sub-periods before and after publication, correlation | `scripts/run_ml.py` |

## Figures

| File | Content | Written by |
|---|---|---|
| [`noise_area_example.png`](noise_area_example.png) | Noise Area, VWAP and trades on one example day | `run_backtest.py` |
| [`equity_curves.png`](equity_curves.png) | Equity curves of the paper variants vs. SPY, full sample | `run_backtest.py` |
| [`metrics_train_vs_test.png`](metrics_train_vs_test.png) | Sharpe ratio, annualized return and volatility of the paper variants, train vs. test | `run_backtest.py` |
| [`cost_sensitivity.png`](cost_sensitivity.png) | Test Sharpe ratio as a function of the cost per share | `run_backtest.py` |
| [`metrics_overview.png`](metrics_overview.png) | Sharpe ratio, return, volatility and max drawdown of **all** implementations, train vs. test | `run_ml.py` |
| [`risk_return.png`](risk_return.png) | Annualized return vs. volatility of all implementations | `run_ml.py` |
| [`ml_equity_test.png`](ml_equity_test.png) | Equity curves in the test period: ML strategy vs. paper rule vs. SPY | `run_ml.py` |

## Raw data

[`tables/`](tables) holds the same results as CSV files with full precision, plus the cross-validation table, the model coefficients and the lookback × volatility-multiplier grid.
