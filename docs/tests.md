# Tests

The test suite has 26 tests in `tests/`. It runs with `pytest` in about two seconds and needs no market data or API key. GitHub Actions runs it on every push (Python 3.10 with pandas 2 and Python 3.12 with pandas 3).

```bash
pytest          # all tests
pytest -q tests/test_backtest.py     # one file
```

## Idea

A backtest can produce a good-looking result for the wrong reason. The two most common reasons are **look-ahead** (a signal uses information that was not yet known) and **wrong accounting** (P&L or costs are computed incorrectly). The tests are built to catch exactly these two.

They use two kinds of artificial data, so every expected value can be computed by hand:

- **Constructed days** (`tests/conftest.py`). `bars_from_paths` turns a price path into minute bars. `quiet_day` is a day that wiggles by ±0.02% and ends where it started. Twenty quiet days give a very narrow Noise Area, so the following day can be designed freely: a straight +1% trend, a breakout that reverses, and so on.
- **Random-walk bars** (`synthetic.make_minute_bars`) with a fixed seed, for tests that compare against a manual computation.

Two test patterns appear several times:

- **Shock test for look-ahead.** Compute a quantity, change prices *after* a point in time, and compute it again. Everything up to that point must be unchanged.
- **Hand computation.** Compute the expected number directly from the definition and compare.

## `test_signals.py` – Noise Area and volatility (4 tests)

| Test | What it checks |
|---|---|
| `test_sigma_has_no_look_ahead` | Prices on day 30 are changed. σ on days ≤ 30 stays the same and σ on day 31 changes. So the Noise Area of a day uses only earlier days. |
| `test_sigma_is_mean_absolute_move_of_previous_days` | σ at one day and minute equals the mean of \|close / open − 1\| over the previous 14 days, computed by hand. |
| `test_gap_adjustment_widens_band_on_the_gap_side` | The upper band is anchored at max(open, previous close) and the lower band at min(open, previous close). |
| `test_daily_vol_uses_only_past_returns` | The volatility used for sizing on day 20 equals the standard deviation of the 14 daily returns before day 20. |

## `test_data.py` – data preparation (6 tests)

| Test | What it checks |
|---|---|
| `test_shapes_and_time_convention` | The data is a days × 390 matrix. Column *j* is the bar starting at 09:30 + *j* minutes, and the execution price of a decision is the open of the next bar. |
| `test_extended_hours_are_dropped` | Pre-market bars are ignored. The day's open is the 09:30 open. |
| `test_vwap_matches_manual_computation` | VWAP equals the cumulative sum of typical price × volume divided by cumulative volume. |
| `test_dividend_is_subtracted_from_previous_close_on_ex_date` | On an ex-dividend date the previous close is reduced by the dividend, and no other day changes. |
| `test_nyse_early_closes` | The half-day calendar (day after Thanksgiving, Christmas Eve, 3 July) is correct for several years. |
| `test_after_hours_bars_on_half_days_are_dropped` | A half day has 210 bars and closes at 13:00, even if the data contains later bars. |

## `test_backtest.py` – trading logic and accounting (6 tests)

| Test | What it checks |
|---|---|
| `test_no_trades_without_breakout` | A quiet day produces no trade and a return of exactly 0. |
| `test_trend_day_gives_one_long_trade_with_exact_pnl` | On a straight +1% day there is exactly one long trade, entered at the first decision time (10:00) and closed at the close. Net P&L equals shares × (exit − entry) − 2 orders × cost per share. |
| `test_vwap_stop_exits_on_reversal_but_opposite_band_holds` | The price rises and then falls back. The base version holds until the close. The band/VWAP stop exits earlier and ends the day with a better return. |
| `test_vol_targeting_caps_leverage` | After a very quiet history the leverage is exactly 4, the cap. |
| `test_costs_reduce_returns` | The same trade earns less with higher costs. |
| `test_minimum_commission_applies_to_small_orders` | A 50-share order pays the \$0.35 minimum per order, not 50 × \$0.0035. |

## `test_metrics.py` – performance metrics (3 tests)

| Test | What it checks |
|---|---|
| `test_sharpe_is_invariant_to_leverage` | Doubling all returns leaves the Sharpe ratio unchanged. |
| `test_annualized_return_of_constant_daily_return` | 252 days of +0.1% give an annualised return of 1.001²⁵² − 1. |
| `test_max_drawdown` | For the returns +10%, −50%, +20% the maximum drawdown is 50%. |

## `test_diagnostics.py` – trade diagnostics (2 tests)

| Test | What it checks |
|---|---|
| `test_false_breakout_is_detected` | A breakout at 10:00 that is back at the open by 10:30 is stopped out at the next decision time and counted as a false breakout. |
| `test_trade_held_to_close_is_not_a_false_breakout` | A trade held until the close is not counted as a false breakout, and it is assigned to the 10:00 entry time. |

## `test_ml.py` – own ML strategy (5 tests)

| Test | What it checks |
|---|---|
| `test_features_have_no_look_ahead` | Prices after the 12:00 decision are changed. The features of the 12:00 row stay the same, and only its label changes. |
| `test_label_is_return_from_execution_to_close` | The label is the return from the execution price (open of the next bar) to the day's close. |
| `test_probs_to_positions_uses_symmetric_margin` | With a margin of 0.03, probabilities 0.60 / 0.52 / 0.50 / 0.48 / 0.40 become long / flat / flat / flat / short. |
| `test_cv_never_trains_on_validation_or_later_years` | In every cross-validation fold the training data ends before the validation year begins. |
| `test_target_positions_reproduce_hold_to_close` | If the model says "long" at every decision time on a trend day, the backtest gives the same single trade and the same P&L as the rule-based strategy. So the ML strategy uses the same execution and cost logic. |

## What the tests do not cover

- They do not check that the strategy is profitable. That is an empirical result (see [`report.md`](report.md)).
- They do not check the downloaded market data itself.
- They use the cost model as specified. Whether \$0.001 slippage per share is realistic is a modelling assumption, examined in the cost sensitivity analysis.
