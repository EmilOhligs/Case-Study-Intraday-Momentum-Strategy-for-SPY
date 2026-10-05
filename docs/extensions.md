# Own Strategy Extensions – Design Document

*Task 3 of the case study: "Invent your own strategy inspired by the paper and evaluate it."*
*Notation follows [`strategy.md`](strategy.md). Status: **design fixed, implementation pending, not yet evaluated.***

---

## 0. Overview

The paper's signal is a single comparison at a single minute: is $P_{t,k}$ outside the Noise Area at $k \in \mathcal K_t$? Both extensions keep this signal and add a layer that decides **whether a breakout is worth trading**.

| | Idea 1: Transient-spike filter | Idea 2: ML meta-labeling |
|---|---|---|
| Weakness addressed | a one-minute spike at HH:00 / HH:30 triggers a trade | not every breakout has the same quality |
| Type | rule-based, 1 parameter | statistical model, ~10 features |
| Acts on | entries only | entries only |
| Main risk | late entries miss part of the move; may be no better than a wider band (VM = 1.5) | overfitting, filtering out the rare big winners |

**Base for comparison:** Extension 1 of the paper (band/VWAP trailing stop, $L_t = 1$). We use 1x exposure so the effect on *signal quality* is not hidden by leverage. All variants are additionally reported with volatility targeting.

---

## 0.1 Evidence from the paper (Section 4 and FAQ)

**Caveat:** all of Section 4 is in-sample (2007–2024) and slices the same data many times (8 patterns, 5 weekdays, 20+ VIX thresholds). We treat these results as **priors for hypotheses**, not as facts, and re-test them on our data. Our data from May 2024 on is out-of-sample even for the paper.

| Paper finding | Implication for us |
|---|---|
| §4.4: Sharpe rises from VM = 1 to VM ≈ 1.5 | ✅ supports Idea 1's premise (weak breakouts are noise). ⚠️ A wider band is the simplest "filter", so **VM = 1.5 (chosen on train) is a mandatory benchmark** for Idea 1. |
| §4.6: edge vanishes at ≈ \$0.03–0.035/share | ✅ every avoided false breakout saves two orders. The filter should help most under conservative costs. |
| §4.5: strategy return vs. 5-day RSI, β = −3.25, p = 0.001 (dealer-gamma proxy) | ✅ strongest result in §4, so RSI is a key feature for Idea 2 |
| §4.1: Sharpe increases with VIX at the open | ✅ add **VIX at open** as a feature (free daily data from CBOE) |
| §4.2: NR4 / NR7 days earn 22 / 16 bp vs. 12 bp unconditional (t = 5.1) | ➕ add a **range-compression** feature (volatility contraction precedes breakouts) |
| §4.3: Wednesday is the best day, probably FOMC | ➕ add an **FOMC-day** flag (economic reason) instead of weekday dummies (data-mining risk) |
| FAQ Q18: trends pause 12:30–14:00 | ✅ supports the time-of-day feature |
| §4.2, FAQ Q19–Q21: hard filters (shorts only at high VIX or below SMA200) **reduce** total return; the unconditional strategy is the most significant (t = 5.34) | ⚠️ conditioning works better for **sizing than for skipping**. For Idea 2, confidence-based sizing becomes the **main variant**, and the random-filter benchmark is essential. |

---

## 1. Pre-registered hypotheses

The hypotheses and success criteria are fixed **before** the test period is looked at:

- **H1 (spike filter):** A confirmation filter reduces the share of false breakouts and increases the test-period Sharpe ratio after costs compared to the base **and** compared to simply widening the band (VM chosen on train, prior from the paper: 1.5).
- **H2 (meta-labeling):** Confidence-based sizing from a meta-model trained only on 2016–2021 trades increases the test-period Sharpe ratio after costs. The hard take/skip version must additionally beat a random filter that skips the same fraction of trades.
- **Success criterion:** a higher test-period Sharpe ratio in **both** cost scenarios. A gain only under the cheap costs, or only in the training period, does not count.

---

## 2. Idea 1 – Transient-spike filter

### 2.1 Motivation

Positions only change at $k \in \{30, 60, \dots\}$. That is the paper's own protection against noise. However, the decision still depends on a single one-minute close. If $P_{t,k}$ is briefly pushed outside the band by a large order, a news headline or a single print, the strategy enters. At the next decision it is often stopped out again: two orders of costs plus a small loss.

**Diagnostic (computed on the training period first):** the false-breakout rate

```math
\mathrm{FBR} = \frac{\#\{\text{entries closed at the next decision time}\}}{\#\{\text{entries}\}},
```

and the average net P&L of these trades. A high FBR with negative average P&L motivates the filter with data, before any filter is tested.

### 2.2 Filter variants

Let $s \in \{+1,-1\}$ be the side of a candidate entry and $B^{s}_{t,k}$ the relevant boundary: $\max(\mathrm{UB}_{t,k}, \mathrm{VWAP}_{t,k})$ for longs and $\min(\mathrm{LB}_{t,k}, \mathrm{VWAP}_{t,k})$ for shorts.

**A. Smoothed price.** Compare the mean of the last $m$ prices with the band:

```math
\bar P^{(m)}_{t,k} = \frac1m \sum_{i=0}^{m-1} P_{t,k-i},
\qquad
\text{enter if } s\,\big(\bar P^{(m)}_{t,k} - B^{s}_{t,k}\big) > 0 .
```

**B. Confirmation.** All of the last $m$ closes must be outside their band:

```math
\text{enter if } \min_{0 \le i < m} \; s\,\big(P_{t,k-i} - B^{s}_{t,k-i}\big) > 0 .
```

**C. Relative volume.** The breakout must come with above-normal volume for this time of day:

```math
\mathrm{RV}_{t,k} = \frac{\sum_{j=k-m}^{k-1} V_{t,j}}{\frac1n \sum_{i=1}^{n} \sum_{j=k-m}^{k-1} V_{t-i,j}},
\qquad
\text{enter if } \mathrm{RV}_{t,k} > \tau_V .
```

The economic reasoning: a real supply/demand imbalance should show up in volume, while a spike on thin volume is noise.

### 2.3 Position rule with filter

The filter applies to **entries only**. Exits keep the paper's fast trailing stop. With $f^{s}_{t,k} \in \{0,1\}$ the filter decision:

```math
x_{t,k} =
\begin{cases}
+1 & \text{if } P_{t,k} > B^{+}_{t,k} \text{ and } \big(x^- = +1 \text{ or } f^{+}_{t,k} = 1\big),\\
-1 & \text{if } P_{t,k} < B^{-}_{t,k} \text{ and } \big(x^- = -1 \text{ or } f^{-}_{t,k} = 1\big),\\
0  & \text{otherwise.}
\end{cases}
```

Rationale for the asymmetry: a delayed entry costs part of the move, while a delayed exit can cost far more.

### 2.4 Parameters and selection

- $m \in \{1, 3, 5, 10\}$ minutes ($m = 1$ is the paper). $\tau_V \in \{1.0, 1.25, 1.5\}$ for variant C.
- Parameters are chosen **on the training period only** by Sharpe after costs. We report the whole grid. A plateau is a good sign, a single peak is a warning.
- Expected trade-off: fewer trades and lower costs vs. later entries.

---

## 3. Idea 2 – ML meta-labeling (own strategy)

### 3.1 Concept

*Meta-labeling* (López de Prado, 2018) separates two questions:

1. **Primary model (the paper's strategy):** which side, long or short? The primary model is unchanged.
2. **Secondary model (ML):** given this signal, how likely is the trade to be profitable? Based on the answer, the trade is taken or skipped (optionally: sized).

The ML model never chooses the direction. It can only veto. This keeps it small, interpretable and hard to overfit compared with a model that predicts returns directly.

### 3.2 Labels

For every entry $\ell$ of the primary strategy in the training period (entry at $(t,k)$, side $s_\ell$):

```math
y_\ell = \mathbf 1\{\Pi^{\text{net}}_\ell > 0\},
```

where $\Pi^{\text{net}}_\ell$ is the trade's P&L after costs (paper cost scenario). Expected sample size: about 1.2–1.8 trades per day × ~1,500 training days ≈ **2,000–2,500 trades**.

### 3.3 Features

All features must be $\mathcal F_{t,k}$-measurable (known at the entry decision). They are signed with the trade side where that makes economic sense.

| # | Feature | Definition | Economic idea |
|---|---|---|---|
| 1 | Breakout strength | $s\,(P_{t,k} - B^{s}_{t,k}) / (O_t\,\sigma_{t,k})$ | strong breakouts are less likely to be noise |
| 2 | VWAP distance | $s\,(P_{t,k} - \mathrm{VWAP}_{t,k}) / P_{t,k}$ | how far the price has already run |
| 3 | Time of day | $k / 390$ | trends pause at lunch and resume after 14:00 (paper FAQ Q18) |
| 4 | Signed gap | $s\,(O_t / \tilde C_{t-1} - 1)$ | breakout in the gap direction vs. against it |
| 5 | 5-day RSI at $t-1$ | standard RSI on daily closes | dealer-gamma proxy (paper §4.5): high RSI means long gamma, so trends are damped |
| 6 | Vol regime | $\hat\sigma^{(5)}_t / \hat\sigma^{(60)}_t$ | is volatility rising or falling? |
| 7 | Relative volume | $\mathrm{RV}_{t,k}$ from §2.2 | participation behind the move |
| 8 | Prior loss today | $\mathbf 1\{\text{a previous trade today lost}\}$ | choppy day indicator |
| 9 | VIX at open | $\mathrm{VIX}_{t,0}$ | Sharpe rises with the VIX (paper §4.1) |
| 10 | Range compression | $(H_{t-1}-L_{t-1}) \,/\, \frac{1}{14}\sum_{i=1}^{14}(H_{t-i}-L_{t-i})$ | NR4/NR7 effect (§4.2): compression precedes breakouts |
| 11 | FOMC day | $\mathbf 1\{t \text{ is an FOMC announcement day}\}$ | Wednesday / FOMC effect (§4.3) |

### 3.4 Model

Logistic regression with L2 penalty on standardised features:

```math
\hat p_\ell = \Pr(y_\ell = 1 \mid \mathbf x_\ell) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol\beta^\top \mathbf x_\ell)}},
\qquad
\min_{\beta}\; -\sum_\ell \Big[y_\ell \ln \hat p_\ell + (1-y_\ell)\ln(1-\hat p_\ell)\Big] + \lambda \lVert\boldsymbol\beta\rVert_2^2 .
```

Why logistic regression and not boosting or a neural net:

- ~2,000 samples and 11 features, i.e. roughly 180 events per parameter. A simple model is appropriate.
- The coefficients are interpretable and can be checked against the economic intuition in §3.3. A wrong sign is a red flag.

### 3.5 Decision rule

**Main variant – confidence-based sizing** (paper §4.2 / FAQ Q19–21 show that hard filters lose total return):

```math
q_\ell = q_t \cdot w(\hat p_\ell), \qquad
w(\hat p) = \min\!\Big(w_{\max},\ \max\!\big(w_{\min},\ \hat p / \bar p\big)\Big),
```

where $\bar p$ is the average predicted probability in training, e.g. $w_{\min} = 0.5$ and $w_{\max} = 1.5$. Weak setups are traded smaller, strong setups larger, and no trade is skipped completely.

**Secondary variant – take/skip:**

```math
\text{take trade } \ell \iff \hat p_\ell > \tau .
```

The threshold $\tau$ (and $w_{\min}, w_{\max}$) are **not** set at 0.5 by default. They are chosen to maximise the Sharpe ratio after costs on the validation folds. The reason: the strategy has a hit ratio around 40% with positive skew, so accuracy is the wrong objective.

### 3.6 Validation (no leakage)

- **Expanding-window cross-validation by year, training period only:** fit on 2016–2017 and validate on 2018, fit on 2016–2018 and validate on 2019, and so on up to 2021. This selects $\lambda$ and $\tau$.
- **Final fit** on all of 2016–2021, then **one** evaluation on the test period 2022–today.
- Features are standardised with means and standard deviations from the training folds only.
- Trades of the same day always stay in the same fold.

### 3.7 Diagnostics we report

- AUC and a calibration plot on the validation folds. Is $\hat p$ informative at all?
- Coefficient signs and their stability across folds.
- **Random-filter benchmark:** skip the same fraction of trades at random (1,000 draws) and show where the meta-model's Sharpe ratio lies in that distribution. This separates "better trades" from "simply fewer trades, lower costs".
- P&L of skipped trades. If the skipped trades contain the large winners, the filter hurts the payoff profile.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Overfitting / too much information** | 11 economically motivated features, each backed by a paper finding (§0.1), L2 penalty, logistic model, hypotheses fixed in advance, test set used once |
| **Filtering out rare big winners** (trend following lives on positive skew) | objective = Sharpe after costs, not accuracy; report the skew and P&L of skipped trades |
| **Late entries** (spike filter) | filter only entries, small $m$; the grid shows the cost of delay |
| **Distribution shift** | skipping a trade can create later entries that never existed in the training data. We monitor the share of such trades; a further step would be iterative re-labeling. |
| **Non-stationarity** (2022+ is a different rate and volatility regime) | expanding-window CV shows whether the coefficients are stable over time; a further step would be yearly walk-forward re-fitting |
| **Multiple testing** | all tested variants are listed in the results table, including the ones that failed |

---

## 5. Evaluation matrix

Same data split, same cost scenarios ($\delta$ = \$0.001 and \$0.005), same metrics (Sharpe, annualised return, annualised volatility, MDD, hit ratio, skew, trades/day).

| Variant | Sizing 1x | Vol targeting |
|---|---|---|
| Ext. 1 (base for comparison) | ✓ | ✓ |
| Ext. 1 with wider band (VM chosen on training, benchmark for the filter) | ✓ | ✓ |
| + spike filter (best $m$ from training) | ✓ | ✓ |
| + meta-labeling, confidence sizing (main) | ✓ | ✓ |
| + meta-labeling, take/skip | ✓ | ✓ |
| + spike filter + meta-labeling | ✓ | ✓ |
| Random filter (same skip rate) | ✓ | – |

---

## 6. Expected edge

- **Spike filter:** fewer false breakouts, so fewer pairs of costly orders. The gain should be larger under the conservative cost scenario.
- **Meta-labeling:** the paper itself shows that the strategy's profitability depends on observable conditions: higher volatility, low RSI (dealer short gamma), time of day. A model that conditions on these should skip a disproportionate share of losing trades in calm, long-gamma, lunchtime conditions.
- **Honest prior:** with only ~4 test years, the Sharpe difference between variants will have wide confidence intervals. An improvement of 0.2–0.3 in Sharpe is not statistically significant on its own. The value lies in a consistent effect across both cost scenarios and both sizing schemes.

---

## 7. Further ideas (out of scope, "further improvements")

| Idea | Evidence | Comment |
|---|---|---|
| **Multi-asset portfolio** (QQQ, IWM, DIA, GLD, …) | FAQ Q12/Q13: diversified Sharpe > 2 (ETFs) and 1.65 (33 futures) | probably the largest gain: more independent bets, same code. Choose assets ex ante by liquidity to avoid selection bias. |
| **Regime-aware sizing** (RSI, VIX) instead of pure vol targeting | §4.1, §4.5 | Vol targeting *cuts* exposure when volatility is high, but the Sharpe is *higher* then. Kelly-type sizing $f \propto \mu/\sigma^2$ would use the expected edge, not only the risk. |
| **Longer lookback** (e.g. 90 days) | FAQ Q6: Sharpe 1.50 vs. 1.35 | only via the training grid, otherwise overfitting |
| **Earlier first decision** (09:45) on Mondays or gap days | §4.3: Mondays miss the first-30-minute move | close to weekday data-mining; needs an economic argument |

---

## 8. Implementation plan

- [ ] `filters.py`: smoothed price, confirmation, relative volume; plug into `backtest.py` via an `entry_filter` config option
- [ ] VM grid on the training period (benchmark for the filter)
- [ ] Diagnostic: false-breakout rate on the training period
- [ ] `features.py`: the 11 features, each with a no-look-ahead unit test (VIX and FOMC dates as small extra data files)
- [ ] `meta.py`: labels, expanding-window CV, logistic regression (NumPy implementation or scikit-learn), sizing weights and threshold choice
- [ ] Random-filter benchmark
- [ ] `scripts/run_extensions.py`: evaluation matrix → `results/extensions_summary.md` and plots
- [ ] Results and discussion → [`report.md`](report.md) §8
