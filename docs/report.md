# Case Study Report – Intraday Momentum Strategy for SPY

*WUTIS Algorithmic Trading Superday · Emil Ohligs*
*Mathematical specification: [`strategy.md`](strategy.md) · Code: [`src/intraday_momentum/`](../src/intraday_momentum)*

> **Status:** implementation and tests complete. Results sections are filled in after the run on real data (marked *TBD*).

---

## 1. Objective and scope

| Task (case study) | Where it is covered |
|---|---|
| 1. Understand the paper: hypothesis, signal, execution, risk controls, regimes | [`strategy.md`](strategy.md), §2–3 below |
| 2. Implement & backtest (base + one extension), train/test split, realistic costs, Sharpe / return / volatility | `backtest.py`, `scripts/run_backtest.py`, §6–7 |
| 3. Own strategy variation, same split / costs / metrics | [`extensions.md`](extensions.md), §8 (*TBD*) |

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
- On half days (13:00 close) the minutes after the close stay empty, and the position is closed at the last bar.
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
| Costs | `max($0.35, $0.0035·q) + δ·q` per order, δ ∈ {0.001, 0.005} | IBKR entry tier. δ = 0.005 is about half the SPY spread. |
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
| Slippage | **none in the code** | $0.001 (paper text) and $0.005 (conservative) | ⚠️ the reference code is more optimistic than the paper |
| Sample | Polygon, ~2 years, no split | Alpaca 2016–today, train/test split | ➕ extension |

**Findings:**

1. The core logic (σ, bands, VWAP condition, decision times) matches exactly.
2. The cross-check found **two omissions in my first version**: the dividend adjustment and the minimum commission. Both are now implemented and unit-tested.
3. The **reference code applies no slippage**, although the paper text states $0.001/share. Results reproduced with the reference code are therefore slightly optimistic. We report both cost scenarios.
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
| `test_sharpe_is_invariant_to_leverage`, `test_alpha_beta_recovers_known_coefficients`, … | the metrics are correct |

CI: GitHub Actions runs the full suite on every push (`.github/workflows/tests.yml`).

---

## 7. Results – replication *(TBD after the data run)*

### 7.1 Paper reference values (2007-05 – 2024-04, paper costs)

| Variant | Total return | Ann. return | Ann. vol | Sharpe | MDD | Hit ratio |
|---|---|---|---|---|---|---|
| Base (opposite band) | 178% | 6.2% | 10.9% | 0.61 | 21% | 54% |
| + band/VWAP stop | 380% | 9.7% | 7.7% | 1.24 | 12% | 43% |
| + vol targeting | 1,985% | 19.6% | 14.3% | 1.33 | 25% | 43% |
| SPY buy & hold | 227% | 7.2% | 20.2% | 0.45 | 56% | 54% |

### 7.2 This replication – train vs. test

*TBD – generated by `scripts/run_backtest.py` → `results/summary.md`, `results/metrics_train_vs_test.png`*

### 7.3 Robustness (train period only) and cost sensitivity (test period)

*TBD – `results/robustness_train_sharpe.csv`, `results/cost_sensitivity.png`*

---

## 8. Own strategy *(TBD)*

Design, pre-registered hypotheses and evaluation plan: see [`extensions.md`](extensions.md) (transient-spike filter + ML meta-labeling). Results follow here.

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
