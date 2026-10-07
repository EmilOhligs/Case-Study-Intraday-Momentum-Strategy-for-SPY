# Own Strategy – ML Long/Short/Flat Model

*Task 3 of the case study: "Invent your own strategy inspired by the paper and evaluate it."*
*Full mathematical specification: [`strategy.md`](strategy.md) Part II (§9–12).*
*Code: [`features.py`](../src/intraday_momentum/features.py), [`ml_strategy.py`](../src/intraday_momentum/ml_strategy.py), [`scripts/run_ml.py`](../scripts/run_ml.py), notebook §8. Notation as in [`strategy.md`](strategy.md).*

---

## 1. Idea and choice of approach

The case study lists "an ML-based model that predicts long/short/flat intraday exposure" as an example of an own variation. We follow that route. A small, interpretable model **replaces** the paper's decision rule, and the paper's insights enter as **features**.

| Kept from the paper | Changed |
|---|---|
| decision grid (every 30 min from 10:00), always flat at the close | **decision rule:** a logistic regression instead of "price outside the Noise Area" |
| execution at the next bar's open, identical costs and sizing options | **stop logic:** the position is re-evaluated every 30 min; if the model turns neutral, the trade is closed |
| the Noise-Area σ as the yardstick for a "normal" move | the breakout is used as a **continuous** strength, not as a yes/no threshold |

Because everything except the decision rule is identical, the comparison with the paper rule isolates the value of the decision rule.

**Alternatives considered** (see §8): a transient-spike filter and ML meta-labeling on top of the paper's signal. We chose the stand-alone model because it is the more original variation and tests the paper's hypothesis directly: if intraday momentum exists, the model should learn it from the data.

---

## 2. Hypotheses (fixed before the test period was evaluated)

- **H1:** The model has predictive power out of sample: test AUC > 0.5.
- **H2:** The ML strategy reaches a **higher test Sharpe ratio after costs than the paper rule (Ext. 1)**.

---

## 3. Model

**Observations.** One row per day $t$ and decision time $k \in \{30, 60, \dots, 360\}$ (10:00–15:30), about 31,600 rows in total.

**Label.** Does the price rise from the execution price to the close?

```math
y_{t,k} = \mathbf 1\{\,C_t / \hat P_{t,k} - 1 > 0\,\}, \qquad \hat P_{t,k} = O_{t,k}\ \text{(next bar's open)} .
```

**Model.** Logistic regression on standardised features with an L2 penalty:

```math
\hat p_{t,k} = \Pr(y_{t,k}=1 \mid \mathbf x_{t,k}) = \frac{1}{1+e^{-(\beta_0+\boldsymbol\beta^\top \mathbf x_{t,k})}},
\qquad
\min_\beta\; -\sum \big[y\ln\hat p + (1-y)\ln(1-\hat p)\big] + \tfrac{1}{2C}\lVert\boldsymbol\beta\rVert_2^2 .
```

**Position rule** with a no-trade zone (margin) $m$:

```math
x_{t,k} = \begin{cases} +1 & \hat p_{t,k} > 0.5 + m \\ -1 & \hat p_{t,k} < 0.5 - m \\ \ \ 0 & \text{otherwise.} \end{cases}
```

The margin controls the trading frequency and therefore the costs.

**Sizing.** Two variants, as in the paper: 100% of equity ("1x") or volatility targeting (2% daily, leverage ≤ 4).

Why logistic regression:

- With an AUC that is only slightly above 0.5, the data are very noisy. A low-variance linear model is the honest first choice.
- The coefficients can be checked against the economic intuition.
- It runs in seconds, so the full cross-validation is cheap.

---

## 4. Features (all known at the decision time; unit-tested)

| Feature | Definition | Inspired by |
|---|---|---|
| `move_sigma` | $(P_{t,k}/O_t - 1)\,/\,\sigma_{t,k}$ | **the paper's signal**, as a continuous breakout strength |
| `vwap_sigma` | $(P_{t,k}/\mathrm{VWAP}_{t,k} - 1)\,/\,\sigma_{t,k}$ | VWAP trailing stop |
| `gap_z` | $(O_t/\tilde C_{t-1} - 1)\,/\,\hat\sigma^{(14)}_t$ | gap adjustment |
| `ret30_z` | return over the last 30 minutes $/\,\hat\sigma^{(14)}_t$ | short-term momentum |
| `time_of_day` | $k/390$ | intraday seasonality (FAQ Q18) |
| `rsi5` | 5-day RSI at $t-1$, rescaled to $[-1, 1]$ | dealer-gamma proxy (§4.5) |
| `vol_regime` | $\ln(\hat\sigma^{(5)}_t / \hat\sigma^{(60)}_t)$ | volatility regimes (§4.1) |

Here $\sigma_{t,k}$ is the Noise-Area σ (previous 14 days only) and $\hat\sigma^{(n)}_t$ is the std of the last $n$ daily returns up to $t-1$. `tests/test_ml.py` verifies that changing prices *after* a decision time leaves that row's features unchanged.

---

## 5. Validation protocol (no test-set information is used)

1. **Expanding-window cross-validation by year on the training period:** fit on 2016…(Y−1) and validate on Y, for Y = 2018, 2019, 2020, 2021.
2. **Grid:** $C \in \{0.01, 0.1, 1\}$ and margin $m \in \{0, 0.01, 0.02, 0.04\}$.
   **Criterion:** mean validation **Sharpe after costs**. Accuracy is not used, because the trading result is what matters.
3. **Feature-set choice, also inside the CV:**
   - A = the 7 features above
   - B = A + the paper's discrete signal (+1/0/−1)

   B would let the model nest the paper rule.
4. **Final fit** on all of 2016–2021 with the selected settings, then **one** evaluation on the test period 2022-01 – 2026-10.

---

## 6. Results

### 6.1 Model selection (training period only)

| Feature set | Best C | Best margin | Mean val. Sharpe | Mean val. AUC |
|---|---|---|---|---|
| **A: 7 features** | 0.01 | 0.02 | **0.53** | 0.532 |
| B: A + paper signal | 1.0 | 0.01 | 0.35 | 0.529 |

Adding the paper's discrete signal does **not** help in CV. The continuous features already carry the information, and B trades more.

Results by validation year for the selected model (A, C = 0.01, m = 0.02):

| Year | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|
| Sharpe | 1.11 | 0.23 | −0.28 | 1.05 |
| AUC | 0.503 | 0.516 | 0.536 | 0.573 |

### 6.2 What the model learned (standardised coefficients)

| Feature | Coef. | Interpretation |
|---|---|---|
| `vol_regime` | −0.148 | rising volatility makes a down-move into the close more likely (volatility feedback) |
| `vwap_sigma` | +0.079 | trading above VWAP predicts continuation. **Supports the paper's VWAP logic.** |
| `time_of_day` | −0.059 | the later in the day, the lower P(up); this partly captures the positive drift being shorter |
| `move_sigma` | −0.058 | *given* the VWAP distance, an extended move from the open tends to fade slightly (the two features are correlated) |
| `rsi5` | +0.031 | after up-days, up-moves into the close are slightly more likely |
| `gap_z`, `ret30_z` | ≈ 0 | no additional information |

**AUC:** 0.554 in-sample (train), **0.518 on the test period**. The predictive power is small but above 0.5 (H1 ✓).

### 6.3 Test period vs. the paper rule (same split, costs and metrics)

| Strategy (test 2022-01 – 2026-10) | Ann. return | Ann. vol | Sharpe | Max DD | Beta |
|---|---|---|---|---|---|
| Ext. 1: band/VWAP stop (1x) | 6.7% | 6.6% | **1.02** | 10.0% | 0.00 |
| Ext. 2: + vol targeting | 15.8% | 14.2% | **1.10** | 22.3% | −0.04 |
| **ML: logistic (1x)** | 5.7% | 10.3% | **0.59** | 12.8% | −0.03 |
| **ML: logistic + vol targeting** | 13.3% | 20.6% | **0.71** | 20.5% | 0.07 |
| SPY buy & hold | 12.2% | 17.3% | 0.75 | 24.5% | – |

In the test period the model is long 45% of the time, short 17% and flat 38% (at decision times), with 1.3 round trips per day.

**H2 is rejected:** over the full test period the ML strategy has a lower Sharpe ratio than the paper rule.

### 6.4 Sub-periods and diversification

| Period (paper costs) | Ext. 1 Sharpe | ML (1x) Sharpe | SPY Sharpe |
|---|---|---|---|
| Test before publication, 2022-01 – 2024-04 | **1.86** | 0.67 | 0.30 |
| After publication, 2024-05 – 2026-10 | 0.03 | **0.51** | 1.26 |

- The paper rule's edge disappears after publication. The ML strategy keeps a modest positive Sharpe ratio in both sub-periods.
- **The correlation of daily returns between ML and Ext. 1 is −0.03** (test period). An inverse-volatility 61/39 mix of Ext. 1 and ML would have had a test Sharpe of about **1.15**. *Descriptive only:* the weights were not chosen on the training period, so this is a hypothesis for further work, not a result.

---

## 7. Assessment

- **Roughly right rather than precisely wrong:** a simple, regularised model has a small but real edge out of sample. It does not beat a well-designed rule on the full test period.
- **Why the rule is hard to beat:** its threshold logic and trailing stop create a positively skewed payoff (skew +1.7). The logistic model trades on small probability differences, with a hit ratio around 51% and negative skew (−1.35). Its gains are more symmetric, its losses less cut.
- **Why it is still interesting:** it is almost uncorrelated with the rule and held up better after the paper was published. Combining both is the natural next step.
- **Overfitting check:** the in-sample (train) Sharpe ratio is low (0.02) while the test Sharpe ratio is 0.59. This is the opposite of an overfitted model. The strong regularisation (C = 0.01) chosen by CV keeps the model simple.

## 8. Further improvements

1. **Combination with the rule:** choose weights on the training period, e.g. inverse volatility. The near-zero correlation is the strongest argument for this.
2. **Label design:** predict the risk-adjusted return, or the return to the next decision time instead of the close. Remove the market drift from the label so the model does not learn a long bias.
3. **Sizing by confidence** instead of the hard ±1 / 0 rule, e.g. position ∝ $\hat p - 0.5$.
4. **Nonlinear models** (shallow gradient boosting) with the same CV protocol. Breakouts are threshold effects, which a linear model can only approximate.
5. **More information:** VIX at the open, FOMC calendar, range compression (NR4/NR7). All three are motivated by paper §4.
6. **Walk-forward re-fitting** every year instead of one fit on 2016–2021.
7. **Multi-asset** (QQQ, IWM, …): more independent bets (paper FAQ Q12/13).

### Alternatives considered (not implemented)

| Idea | Why it was not the main route |
|---|---|
| **Transient-spike filter** (require the breakout for *m* minutes, or with above-normal volume) | an extension, not an own strategy. Its premise is supported by the data: 34% of Ext. 1 entries are stopped out at the next decision, at −17 bp vs. +12.5 bp for the other trades (train period). Benchmark would be a wider band (VM = 1.5). |
| **ML meta-labeling** (model vetoes or sizes the paper's trades) | depends on the paper's signal. The paper's FAQ (Q19–21) shows that hard filters often cost total return. |

**Evidence from the paper (§4, FAQ) used in the design:** RSI as a gamma proxy (§4.5, p = 0.001) → `rsi5`. Higher Sharpe at higher volatility (§4.1) → `vol_regime`. Intraday seasonality (FAQ Q18) → `time_of_day`. VWAP as the better stop (§3, FAQ Q22) → `vwap_sigma`. All of §4 is in-sample in the paper, so we only used it as a source of hypotheses.
