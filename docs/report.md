# Case Study Report – Intraday Momentum Strategy for SPY

*WUTIS Algorithmic Trading Superday · Emil Ohligs*
*Mathematical specification: [`strategy.md`](strategy.md) · Code: [`src/intraday_momentum/`](../src/intraday_momentum)*

> **Status:** complete. Paper replication and own ML strategy evaluated on SPY 1-minute data 2016-01 – 2026-10.

---

## 1. Objective and scope

| Task (case study) | Where it is covered |
|---|---|
| 1. Understand the paper: hypothesis, signal, execution, risk controls, regimes | §1.1 below, [`strategy.md`](strategy.md) |
| 2. Implement & backtest (base + one extension), train/test split, realistic costs, Sharpe / return / volatility | `backtest.py`, `scripts/run_backtest.py`, §6–7 |
| 3. Own strategy variation, same split / costs / metrics | §8 below, details in [`extensions.md`](extensions.md) |

### 1.1 The paper in brief

| | |
|---|---|
| **Core hypothesis** | Intraday trends come from persistent supply/demand imbalances. If the move from the open is abnormally large for the time of day, it tends to continue (time-series momentum). |
| **Signal** | Noise Area = average absolute move from the open at each minute over the last 14 days, anchored at max/min(open, previous close). Long above the upper band, short below the lower band. |
| **Execution** | Decisions only at HH:00 / HH:30 from 10:00, flat at the close, \$0.0035/share commission + \$0.001 slippage. |
| **Risk controls** | (1) Trailing stop at max(UB, VWAP) / min(LB, VWAP). (2) Volatility targeting: 2% daily, leverage ≤ 4. (3) No overnight positions. |
| **Why it may work** | Under-reaction to news (disposition effect, limited attention, slow-moving capital). Order flow of institutions executing over the day. Dealer delta-hedging when gamma is short amplifies moves. Positive skew from cut losers and held winners. |
| **When it fails** | Quiet, range-bound markets (many false breakouts, e.g. 2016–17). Sharp V-shaped reversals (news; March 2020; April 2025). Long dealer gamma (damped trends). Costs above ~\$0.03/share. Crowding / alpha decay after publication. |

---

## 2. Data

| Item | Choice | Reason |
|---|---|---|
| Instrument | SPY | as in the paper |
| Source | Alpaca Market Data API, SIP (consolidated tape) | free and covers all exchanges. The paper used IQFeed, which is paid. |
| Frequency | 1-minute OHLCV bars | as in the paper |
| Period | 2016-01 – today | earliest data available on the free plan. The paper covers 2007-05 – 2024-04. |
| Prices | unadjusted (raw) for the strategy | commission is charged per share, so the real price level matters |
| Benchmark | SPY buy & hold, split- and dividend-adjusted daily closes | total return, i.e. a fair comparison |
| Dividends | derived from raw vs. adjusted daily closes (`D_t = C_{t-1}^{raw}·(1 − f_{t-1}/f_t)`) | needed for the gap adjustment (§4) |

**Cleaning rules (`data.py`):**

- Only the regular session (09:30–16:00 ET) is kept. Pre-market and after-hours bars are dropped.
- Missing minutes inside the session are forward-filled: no trade in that minute means the price is unchanged.
- On half days (13:00 close: Jul 3, day after Thanksgiving, Dec 24) the vendor's after-hours bars from 13:00 on are removed via an NYSE early-close calendar, and the position is closed at 13:00. A first version missed this, and the check on real data found it (22 half days now).
- "Dividends" below \$0.05 are rounding noise from adjusted vs. raw prices and are ignored.
- Days with fewer than 180 bars are dropped.

---

## 3. Implementation decisions

| Decision | Choice | Rationale |
|---|---|---|
| Time convention | bar labelled 09:59 = price at 10:00 | documented in `data.py` and unit-tested |
| Decision → execution | signal from the close at minute *k*, execution at the **open of the next bar** | rules out look-ahead bias by construction |
| Noise Area σ | mean of \|P/O − 1\| over the previous 14 days, per minute of the day | paper Eq. (σ). Still valid if up to ~20% of the window is missing (half days). |
| Stop variants | Base: opposite band (stateful). Ext. 1: max/min(band, VWAP) (stateless) | paper Tables 1 and 2 |
| Sizing | `floor(AUM · L / Open)`, with L = 1 or `min(4, 2% / σ̂)` | paper Table 3 |
| Costs | `max($0.35, $0.0035·q) + $0.001·q` per order | IBKR entry-tier commission plus the paper's slippage estimate |
| Risk-free rate | 0 in the Sharpe ratio | same as the paper. With 2022–2025 T-bill rates around 4–5%, the excess-return Sharpe would be lower. |
| Train / test | 2016–2021 / 2022–today, each starting with $100k | parameters are fixed to the paper's values, so nothing is fitted on the test set |

---

## 4. Cross-check with the authors' reference code

The authors publish a Python version of their backtest on [concretumgroup.com](https://www.concretumgroup.com/python-backtesting-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/), linked in the paper's FAQ (Q1). **This repo was written independently from the paper and then compared against that code.**

| Aspect | Reference code | This repo | Status |
|---|---|---|---|
| Noise Area σ | 14-day rolling mean of \|close/open − 1\| per minute (min. 13 obs.), shifted one day | same (min. 12 obs.) | ✅ identical logic |
| Bands | `max(open, prev_close_adj)·(1+VM·σ)`, `min(...)·(1−VM·σ)` | same | ✅ |
| Dividend adjustment of previous close | `prev_close − dividend` on ex-date | **was missing → added** | 🔧 fixed after the cross-check |
| Signal | long if close > UB **and** > VWAP. Short if close < LB **and** < VWAP. Else flat. | same (`stop="band_vwap"`) | ✅ |
| Decision times | `min_from_open % 30 == 0`, with `min_from_open` = bar label − 09:30 + 1 → uses the 09:59 bar at "10:00" | column 29 = 09:59 bar | ✅ identical |
| Execution | exposure shifted by one bar and multiplied by the next 1-minute return → entry at the close of the decision bar | entry at the open of the next bar | ≈ same price, ours is slightly stricter |
| Vol targeting | `min(4, 0.02 / spx_vol)`, `spx_vol` over **15** days | **14** days, as stated in the paper | ⚠️ deliberate deviation (paper text) |
| Share rounding | `round()` | `floor()` | ⚠️ minor. `floor` never exceeds the available capital. |
| Commission | `max($0.35, 0.0035·q)` per order, a reversal counts as 2 | **minimum was missing → added**, reversal = 2 orders | 🔧 fixed after the cross-check |
| Slippage | **none in the code** | $0.001 (paper text) | ⚠️ the reference code is more optimistic than the paper |
| Sample | Polygon, ~2 years, no split | Alpaca 2016–today, train/test split | ➕ extension |

**Findings:**

1. The core logic (σ, bands, VWAP condition, decision times) matches exactly.
2. The cross-check found **two omissions in my first version**: the dividend adjustment and the minimum commission. Both are now implemented and unit-tested.
3. The **reference code applies no slippage**, although the paper text states $0.001/share. Results reproduced with the reference code are therefore slightly optimistic. We apply the paper's \$0.001.
4. There are small inconsistencies between the paper and the code: the vol window is 15 days in the code vs. 14 in the paper. We follow the paper.

---

## 5. Differences to the paper (and their expected effect)

| Difference | Expected effect on results |
|---|---|
| Sample 2016– instead of 2007– (2008 is missing, the strategy's best year: +63%) | lower long-run averages than the paper |
| Alpaca SIP vs. IQFeed | small differences in individual minute bars, negligible for 30-minute decisions |
| No lower commission tier ($0.002 above 300k shares/month, about 20% of days in the paper) | slightly higher costs than the paper |
| Python vs. Matlab | none expected |
| Execution at the next bar's open | at most one minute later than the paper. Slightly conservative. |

---

## 6. Validation (unit tests, `pytest`)

All 26 tests are explained in [`tests.md`](tests.md). The most important ones:

| Test | What it guarantees |
|---|---|
| `test_sigma_has_no_look_ahead` | Changing prices on day *k* leaves σ on days ≤ *k* unchanged. **No look-ahead.** |
| `test_sigma_is_mean_absolute_move_of_previous_days` | σ matches a hand computation |
| `test_gap_adjustment_widens_band_on_the_gap_side` | the bands follow Eq. (3)–(4) exactly |
| `test_daily_vol_uses_only_past_returns` | the vol-targeting input uses only past data |
| `test_dividend_is_subtracted_from_previous_close_on_ex_date` | the dividend adjustment works |
| `test_shapes_and_time_convention` | bar ↔ time mapping and execution price are correct |
| `test_vwap_matches_manual_computation` | VWAP is correct |
| `test_trend_day_gives_one_long_trade_with_exact_pnl` | P&L and costs are exact to the cent on a constructed trend day |
| `test_vwap_stop_exits_on_reversal_but_opposite_band_holds` | the two stop rules behave as specified |
| `test_minimum_commission_applies_to_small_orders` | the $0.35 minimum is applied |
| `test_vol_targeting_caps_leverage` | leverage ≤ 4 |
| `test_sharpe_is_invariant_to_leverage`, … | the metrics are correct |
| `test_nyse_early_closes`, `test_after_hours_bars_on_half_days_are_dropped` | half days end at 13:00 |
| `test_false_breakout_is_detected`, … | the trade diagnostics are correct |
| `test_features_have_no_look_ahead` (ML) | changing prices after a decision time leaves that row's features unchanged |
| `test_cv_never_trains_on_validation_or_later_years` (ML) | the expanding-window CV has no leakage |
| `test_target_positions_reproduce_hold_to_close` (ML) | ML positions run through the same engine with exact P&L |

CI: GitHub Actions runs the full suite on every push (`.github/workflows/tests.yml`).

---

## 7. Results – replication

All numbers come from `scripts/run_backtest.py` / the notebook (`results/summary.md`). The test period is 2022-01-03 – 2026-10-02.

### 7.1 Replication check against the paper

Yearly returns of the full model (Ext. 2) vs. the paper's monthly table (FAQ Q4/Q24):

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| This repo | −13.4% | −9.3% | 55.6% | 5.7% | 25.3% | 29.7% | 25.1% | 39.2% | 32.6% |
| Paper | −12.8% | −6.9% | 61.1% | 6.9% | 26.8% | 34.8% | 24.4% | 37.2% | 32.2% |

The **correlation is 0.99**. Volatility (14.2–14.6% vs. 14.3%), max drawdown (22–25% vs. 25%), hit ratio (42–46% vs. 43%) and positive skew also match. Our 0.86–0.92 round trips per day correspond to the paper's ~1.8 *orders* per day.

### 7.2 Train vs. test

| Strategy | Sharpe train | Sharpe test | Ann. return test | Ann. vol test | MDD test |
|---|---|---|---|---|---|
| Base: opposite-band stop | 0.37 | 0.74 | 6.8% | 9.5% | 8.9% |
| Ext. 1: + band/VWAP stop | 0.66 | 1.02 | 6.7% | 6.6% | 10.0% |
| Ext. 2: + vol targeting | 0.93 | 1.10 | 15.8% | 14.2% | 22.3% |
| SPY buy & hold | 1.05 | 0.75 | 12.2% | 17.3% | 24.5% |

- The paper's ranking (base < VWAP stop < vol targeting) holds in both periods.
- Vol targeting adds return mainly through leverage: on average 2.7x, and capped at 4x on 22% of days. Its Sharpe gain is small.
- Figures: `results/metrics_train_vs_test.png`, `results/equity_curves.png`. All implementations side by side, including the own strategy: `results/metrics_overview.png`, `results/risk_return.png`.

### 7.3 Costs and robustness

- **Cost sensitivity (test):** the Ext. 1 Sharpe falls from 1.03 (\$0.0035/share) to 0.76 (\$0.0235/share). The edge survives realistic costs (`results/cost_sensitivity.png`).
- **Parameter grid (train only, Ext. 2):** Sharpe ratios between 0.82 and 1.23. The maximum is at VM = 1.5 / lookback 14, the same as the paper's §4.4. The grid is noisy, with no clean plateau, so parameter choice moves the Sharpe by ±0.2.

### 7.4 After publication

| Ext. 2, paper costs | Ann. return | Sharpe |
|---|---|---|
| Test before publication (2022-01 – 2024-04) | 33.7% | 2.17 |
| After publication (2024-05 – 2026-10) | 0.9% | 0.13 |
| SPY, after publication | 20.8% | 1.26 |

All three variants have a Sharpe ratio of about 0 after the paper's publication. The cause could be alpha decay (crowding) or a regime effect (calm, rising market; 0DTE options). With 2.4 years (t ≈ 0.2) the two cannot be separated statistically.

### 7.5 Trade-level diagnostics (train period, Ext. 1)

- **False breakouts:** 34% of entries are stopped out at the very next decision time. These average −17 bp, all other trades +12.5 bp.
- **Lunch time:** entries between 12:00 and 14:00 have a negative average return. Entries from 14:30 on are positive. This is consistent with paper FAQ Q18.
- **Volatility regimes** (quintiles of trailing realised vol): the relationship is not monotonic. The strategy is weak in very calm years (2016–17) and in V-shaped crashes (2020, 2025), and strong in trending volatile years (2018, 2022).

---

## 8. Own strategy – ML long/short/flat model

Full design and discussion: [`extensions.md`](extensions.md).

- **What:** a logistic regression on 7 paper-inspired features predicts P(up until the close) at every decision time. The position is long, short or flat with a no-trade margin. Execution, costs and sizing are identical to the replication.
- **Selection:** the feature set, C and margin are chosen by expanding-window CV by year on 2016–2021 only (selected: 7 features, C = 0.01, margin = 0.02). Adding the paper's discrete signal did not help in CV.

| Test 2022-01 – 2026-10 | Sharpe | Ann. return | Max DD |
|---|---|---|---|
| Ext. 1 (paper rule, 1x) | 1.02 | 6.7% | 10.0% |
| **ML (1x)** | **0.59** | 5.7% | 12.8% |
| Ext. 2 (paper rule, vol targeting) | 1.10 | 15.8% | 22.3% |
| **ML + vol targeting** | **0.71** | 13.3% | 20.5% |

**Findings:**

- **H1 confirmed:** test AUC 0.518 > 0.5, a small but real out-of-sample edge.
- **H2 rejected:** over the full test period, ML does not beat the rule.
- **After publication:** ML Sharpe 0.51 vs. 0.03 for the rule.
- **Correlation with the rule:** −0.03. This makes a combination of both the most promising next step.
- **Coefficients:** VWAP distance is positive, which supports the paper's VWAP logic. Rising volatility is negative.

---

## 9. Limitations and open risks

- **Financing and borrowing costs are not modelled.** This affects leverage up to 4x and short positions.
- **Test period length.** With about 4 test years, t ≈ SR·√4, so a Sharpe ratio of about 1 is only borderline significant.
- **Regime dependence.** The strategy earns in volatile, trending markets and loses in quiet range-bound years (2016–17 in the paper).
- **Multiple testing.** Every additional variant tested increases the chance of a lucky result. The test set is therefore used only once per variant.
- **Survivorship bias** does not apply to SPY itself. It would apply if single stocks were selected with hindsight, as in the paper's FAQ Q12.

---

## 10. Decision log

| Date | Change |
|---|---|
| 2026-10-04 | Initial implementation: data pipeline, Noise Area, base + 2 extensions, metrics, tests, CI |
| 2026-10-05 | Cross-check with the authors' reference code → added dividend adjustment and minimum commission (+2 tests). Added mathematical specification (`docs/strategy.md`). |
| 2026-10-05 | Own-strategy design (`docs/extensions.md`): spike filter + ML meta-labeling, informed by paper §4 / FAQ (VM benchmark, RSI/VIX/NR/FOMC features, sizing instead of skipping). |
| 2026-10-06 | Real-data run: fixed half-day after-hours bars (NYSE early-close calendar) and dividend noise. Replication validated against the paper's yearly table (corr. 0.99). |
| 2026-10-06 | Own strategy switched from extension ideas to a stand-alone ML model (logistic regression); CV on train, one test evaluation. Results in §8 and `extensions.md`. |
