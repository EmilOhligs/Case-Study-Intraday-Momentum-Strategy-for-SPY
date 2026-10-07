# Intraday Momentum on SPY – Mathematical Specification

*Part I: the paper's strategy (§1–8). Part II: our own ML strategy (§9–12).*

*Based on Zarattini, Aziz & Barbon (2024), "Beat the Market". This document states every step of the strategy and its evaluation as a formula, in the same order as the code in [`src/intraday_momentum/`](../src/intraday_momentum). Each section names the module that implements it.*

**Contents**

1. [Notation](#1-notation)
2. [The Noise Area](#2-the-noise-area-signalspy)
3. [Signal and position](#3-signal-and-position-backtestpy)
4. [Position sizing](#4-position-sizing)
5. [Profit and loss with transaction costs](#5-profit-and-loss-with-transaction-costs)
6. [Performance metrics](#6-performance-metrics-metricspy)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Summary of parameters](#8-summary-of-parameters)
9. [ML: sample, label and features](#9-sample-label-and-features)
10. [ML: model and estimation](#10-model-and-estimation)
11. [ML: model selection and evaluation](#11-model-selection-and-evaluation)
12. [ML: parameters](#12-parameters-of-the-ml-strategy)

---

## 1. Notation

**Days and minutes.** Trading days are indexed by $t = 1,\dots,T$. The regular session (09:30–16:00 ET) consists of $N = 390$ one-minute bars $j = 0,\dots,N-1$. Bar $j$ covers the interval $[09{:}30 + j,\ 09{:}30 + j + 1)$. On half days the session ends earlier. In general $N_t \le N$ denotes the number of bars on day $t$ (`data.py`).

**Prices.** Each bar has open, high, low, close and volume $(O_{t,j}, H_{t,j}, L_{t,j}, C_{t,j}, V_{t,j})$. The price observed $k$ minutes after the open is the close of the bar that ends at $09{:}30+k$:

```math
P_{t,k} := C_{t,k-1}, \qquad k = 1,\dots,N_t .
```

The open and close of the day are

```math
O_t := O_{t,0}, \qquad C_t := P_{t,N_t}.
```

---

## 2. The Noise Area (`signals.py`)

### 2.1 Average move from the open

On each past day $t-i$, the absolute relative move from the open after $k$ minutes is

```math
m_{t-i,k} = \left| \frac{P_{t-i,k}}{O_{t-i}} - 1 \right| .
```

The *size of a normal move* at minute $k$ is the mean over the previous $n = 14$ days (the lookback):

```math
\sigma_{t,k} = \frac{1}{n}\sum_{i=1}^{n} m_{t-i,k} .
```

The sum starts at $i=1$, so $\sigma_{t,k}$ uses only previous days (no look-ahead). In the code this is a rolling mean over days, shifted by one day.

### 2.2 Boundaries with gap adjustment

Let $C_{t-1}$ be the previous close and $D_t$ the cash dividend if $t$ is an ex-dividend date ($D_t = 0$ otherwise). The dividend-adjusted previous close is $\tilde C_{t-1} = C_{t-1} - D_t$. With the volatility multiplier $\mathrm{VM}$ (paper: $\mathrm{VM}=1$):

```math
\begin{aligned}
\mathrm{UB}_{t,k} &= \max\!\big(O_t,\ \tilde C_{t-1}\big)\,\big(1 + \mathrm{VM}\,\sigma_{t,k}\big),\\
\mathrm{LB}_{t,k} &= \min\!\big(O_t,\ \tilde C_{t-1}\big)\,\big(1 - \mathrm{VM}\,\sigma_{t,k}\big).
\end{aligned}
```

The Noise Area is the interval $[\mathrm{LB}_{t,k},\ \mathrm{UB}_{t,k}]$.

### 2.3 Session VWAP

With the typical price $\bar p_{t,j} = (H_{t,j}+L_{t,j}+C_{t,j})/3$:

```math
\mathrm{VWAP}_{t,k} = \frac{\sum_{j=0}^{k-1} \bar p_{t,j}\,V_{t,j}}{\sum_{j=0}^{k-1} V_{t,j}} .
```

It uses only bars up to minute $k$. Mathematically, the VWAP is a weighted mean: the expected price under the normalised volume distribution $V_{t,j}/\sum_i V_{t,i}$.

---

## 3. Signal and position (`backtest.py`)

### 3.1 Decision times

Positions can only change at the decision times

```math
\mathcal K_t = \{30,\ 60,\ 90,\ \dots\} \cap \{1,\dots,N_t-1\},
```

i.e. 10:00, 10:30, …, 15:30 on a full day. Let $x_{t,k} \in \{-1, 0, +1\}$ be the position held after the decision at minute $k$ (short, flat, long). Each day starts flat ($x_{t,0} = 0$). Between decisions the position does not change.

### 3.2 Base model: stop at the opposite band

At $k\in\mathcal K_t$, with $x^-$ the current position:

```math
x_{t,k} =
\begin{cases}
+1 & \text{if } P_{t,k} > \mathrm{UB}_{t,k},\\
-1 & \text{if } P_{t,k} < \mathrm{LB}_{t,k},\\
x^- & \text{otherwise.}
\end{cases}
```

This rule has **memory**: inside the Noise Area the previous position is kept. A long position is closed, and immediately reversed, only when the price crosses the *lower* band.

### 3.3 Extension 1: trailing stop at the current band and VWAP

```math
x_{t,k} =
\begin{cases}
+1 & \text{if } P_{t,k} > \max(\mathrm{UB}_{t,k},\ \mathrm{VWAP}_{t,k}),\\
-1 & \text{if } P_{t,k} < \min(\mathrm{LB}_{t,k},\ \mathrm{VWAP}_{t,k}),\\
0  & \text{otherwise.}
\end{cases}
```

This rule is **memoryless**. As soon as a long falls back below the upper band or below VWAP, it is closed. This is exactly a trailing stop at $\max(\mathrm{UB}_{t,k}, \mathrm{VWAP}_{t,k})$.

### 3.4 Execution

A decision taken at $(t,k)$ is executed at the open of the next bar:

```math
\hat P_{t,k} = O_{t,k} \qquad (\text{the open of the next bar, i.e. after the decision}).
```

Every open position is closed at the day's close $C_t$, i.e. $x_{t,N_t} = 0$. There is no overnight exposure.

---

## 4. Position sizing

The number of shares is fixed at the open of each day:

```math
q_t = \left\lfloor \frac{A_{t-1}\, L_t}{O_t} \right\rfloor ,
```

where $A_{t-1}$ is the equity at the end of day $t-1$ and $L_t$ is the leverage.

**Full exposure (base, Extension 1):** $L_t = 1$.

**Volatility targeting (Extension 2).** With daily close-to-close returns $r_s = C_s/C_{s-1} - 1$, the realised volatility over the previous $n = 14$ days is

```math
\hat\sigma_t = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\big(r_{t-i} - \bar r_t\big)^2},
\qquad \bar r_t = \frac1n\sum_{i=1}^n r_{t-i},
```

which uses only past daily returns. Then

```math
L_t = \min\!\left(L_{\max},\ \frac{\sigma^\star}{\hat\sigma_t}\right),
\qquad \sigma^\star = 2\%,\quad L_{\max} = 4 .
```

Exposure is reduced when the market is volatile and increased when it is calm. SPY's daily volatility is typically around 1%, so the typical leverage is about 2.

---

## 5. Profit and loss with transaction costs

Let the day's round trips be $\ell = 1,\dots,R_t$, each with side $s_\ell\in\{\pm1\}$, entry price $e_\ell$ and exit price $z_\ell$. Every entry, exit or reversal leg is one order of $q_t$ shares. $M_t$ is the number of orders, and a reversal from $+1$ to $-1$ counts as two orders.

**Gross and net P&L:**

```math
\Pi^{\text{gross}}_t = \sum_{\ell=1}^{R_t} s_\ell\, q_t\,(z_\ell - e_\ell),
\qquad
\Pi_t = \Pi^{\text{gross}}_t - M_t\,\kappa(q_t),
```

where the cost of one order is

```math
\kappa(q) = \max\big(\$0.35,\ \$0.0035 \cdot q\big) + \$0.001 \cdot q ,
```

i.e. the Interactive Brokers commission (\$0.0035 per share, at least \$0.35 per order) plus \$0.001 slippage per share, the paper's own estimate.

**Equity and returns:**

```math
A_t = A_{t-1} + \Pi_t, \qquad R^{\text{strat}}_t = \frac{\Pi_t}{A_{t-1}}, \qquad A_0 = \$100{,}000 .
```

---

## 6. Performance metrics (`metrics.py`)

Let $R_1,\dots,R_T$ be daily returns, with mean $\bar R$ and sample standard deviation $s_R$. A year has 252 trading days.

| Metric | Formula |
|---|---|
| Total return | $\prod_{t=1}^T (1+R_t) - 1$ |
| Annualised return (CAGR) | $\big(\prod_{t=1}^T (1+R_t)\big)^{252/T} - 1$ |
| Annualised volatility | $s_R \sqrt{252}$ |
| Sharpe ratio | $\dfrac{\bar R - r_f/252}{s_R}\sqrt{252}$, with $r_f = 0$ in our tables |
| Max drawdown | $\max_t \big(1 - E_t / \max_{u\le t} E_u\big)$, with $E_t = \prod_{u\le t}(1+R_u)$ |
| Hit ratio | share of days with $R_t > 0$ among all days with $R_t \neq 0$ (days with a trade) |

> **Remark – why $\sqrt{252}$.** If daily returns are independent with variance $s^2$, the variance over 252 days is $252\,s^2$. The standard deviation therefore scales with $\sqrt{252}$, while the mean scales with $252$. Hence the Sharpe ratio scales with $252/\sqrt{252} = \sqrt{252}$.

> **Proposition – leverage does not change the Sharpe ratio.** For $r_f = 0$ and any constant $\lambda > 0$: $\mathrm{SR}(\lambda R) = \mathrm{SR}(R)$.
>
> *Proof.* The mean of $\lambda R$ is $\lambda\bar R$ and its standard deviation is $\lambda s_R$, so the ratio is unchanged. ∎
>
> Strategies should therefore be compared by their **Sharpe ratio**, not by total return. Volatility targeting changes $L_t$ over time, so it *can* change the Sharpe ratio. However, most of the increase in total return from Extension 2 comes from the average leverage.

**Alpha and beta.** We regress strategy returns on SPY returns by OLS:

```math
R^{\text{strat}}_t = \alpha + \beta\, R^{\text{SPY}}_t + \varepsilon_t,
\qquad
\hat{\theta} = (X^\top X)^{-1} X^\top y, \quad X = [\mathbf 1,\ R^{\text{SPY}}].
```

The standard errors are $\widehat{\mathrm{se}}(\hat\theta_i) = \sqrt{\hat s^2_\varepsilon\,[(X^\top X)^{-1}]_{ii}}$ with $\hat s^2_\varepsilon = \hat\varepsilon^\top\hat\varepsilon/(T-2)$, and the t-statistics are $t_i = \hat\theta_i / \widehat{\mathrm{se}}(\hat\theta_i)$. We report the annualised alpha $252\,\hat\alpha$.

**Statistical significance of the Sharpe ratio.** The t-statistic of the mean daily return is $t = \bar R / (s_R/\sqrt T) = \mathrm{SR}_{\text{daily}}\sqrt T$. With $T = 252\,Y$ days ($Y$ years) this becomes

```math
t \approx \mathrm{SR}_{\text{ann}}\,\sqrt{Y}.
```

For example, a Sharpe ratio of 1 over 4 test years gives $t \approx 2$, which is only borderline significant.

---

## 7. Evaluation protocol

- **Split.** The sample is split chronologically into a training period (2016–2021) and a test period (2022–today). Each period is simulated separately, starting with $A_0$ = \$100,000.
- **No fitting on the test set.** All variants use the paper's parameters $(n, \mathrm{VM}, \sigma^\star, L_{\max}) = (14, 1, 2\%, 4)$. The grid $n\in\{7,14,30,60\}$, $\mathrm{VM}\in\{0.8, 1.0, 1.2, 1.5\}$ is evaluated **on the training period only**, as a robustness check. A good result is a flat plateau of Sharpe ratios, not a single sharp peak.
- **No look-ahead.** $\sigma_{t,k}$, $\hat\sigma_t$, $\tilde C_{t-1}$ and $L_t$ use only data from previous days. $P_{t,k}$ and $\mathrm{VWAP}_{t,k}$ use only bars up to the decision minute. Execution happens at the open of the next bar. Unit tests check these properties (`tests/test_signals.py`).
- **Costs.** Commission plus \$0.001 slippage per share as in the paper, plus a sensitivity sweep of the total cost per share from \$0.0035 to \$0.0235 in the test period.

---

## 8. Summary of parameters

| Symbol | Meaning | Value |
|---|---|---|
| $n$ | lookback for $\sigma_{t,k}$ and $\hat\sigma_t$ | 14 days |
| $\mathrm{VM}$ | volatility multiplier | 1 |
| $\mathcal K_t$ | decision times | every 30 min from 10:00 |
| $\sigma^\star$ | daily volatility target | 2% |
| $L_{\max}$ | leverage cap | 4 |
| $c$ | commission per share | \$0.0035 |
| $c_{\min}$ | minimum commission per order | \$0.35 |
| – | slippage per share | \$0.001 |
| $A_0$ | initial capital | \$100,000 |

---
---

# Part II – Own strategy: ML long/short/flat model

*Implemented in [`features.py`](../src/intraday_momentum/features.py) and [`ml_strategy.py`](../src/intraday_momentum/ml_strategy.py). Design rationale and results: [`extensions.md`](extensions.md).*

Everything except the **decision rule** is identical to Part I: the decision times $\mathcal K_t$ (§3.1), execution at $O_{t,k}$ (§3.4), flat at the close, sizing (§4) and costs (§5). So this part only specifies how the position $x_{t,k}$ is chosen.

## 9. Sample, label and features

### 9.1 Sample

One observation per day and decision time:

```math
\mathcal D = \{(t,k) : t = 1,\dots,T,\ k \in \mathcal K_t\}, \qquad |\mathcal D| \approx 12 \cdot T .
```

Observations with incomplete features are dropped. This is the burn-in of 60 days for $\hat\sigma^{(60)}$.

### 9.2 Label

The target is whether the price rises from the execution price to the close:

```math
r_{t,k} = \frac{C_t}{\hat P_{t,k}} - 1, \qquad y_{t,k} = \mathbf 1\{r_{t,k} > 0\} \in \{0,1\}.
```

$y_{t,k}$ depends on prices after $(t,k)$. It is only used as a training target, never as an input.

### 9.3 Features

Let $\sigma_{t,k}$ be the Noise-Area σ (§2.1), and let $\hat\sigma^{(n)}_t$ be the sample standard deviation of the daily returns $r_{t-n},\dots,r_{t-1}$ (§4). The feature vector $\mathbf x_{t,k} \in \mathbb R^7$ is

```math
\begin{aligned}
x^{(1)}_{t,k} &= \frac{P_{t,k}/O_t - 1}{\sigma_{t,k}} && \text{move from the open in Noise-Area units}\\
x^{(2)}_{t,k} &= \frac{P_{t,k}/\mathrm{VWAP}_{t,k} - 1}{\sigma_{t,k}} && \text{distance to VWAP}\\
x^{(3)}_{t} &= \frac{O_t/\tilde C_{t-1} - 1}{\hat\sigma^{(14)}_t} && \text{overnight gap (dividend-adjusted)}\\
x^{(4)}_{t,k} &= \frac{P_{t,k}/P_{t,k-30} - 1}{\hat\sigma^{(14)}_t} && \text{last 30 minutes } (P_{t,0} := O_t)\\
x^{(5)}_{k} &= k/390 && \text{time of day}\\
x^{(6)}_{t} &= \big(\mathrm{RSI}^{(5)}_{t-1} - 50\big)/50 && \text{5-day RSI at the previous close, in } [-1,1]\\
x^{(7)}_{t} &= \ln\!\big(\hat\sigma^{(5)}_t / \hat\sigma^{(60)}_t\big) && \text{volatility regime}
\end{aligned}
```

**RSI (Wilder).** With daily price changes $\Delta_s = C_s - C_{s-1}$ and the exponential average $\mathrm{EMA}_\alpha$ with $\alpha = 1/5$:

```math
\mathrm{RS}_s = \frac{\mathrm{EMA}_\alpha\big(\max(\Delta_s,0)\big)}{\mathrm{EMA}_\alpha\big(\max(-\Delta_s,0)\big)},
\qquad
\mathrm{RSI}^{(5)}_s = 100 - \frac{100}{1 + \mathrm{RS}_s}.
```

**No look-ahead.** Every feature only uses information available at the decision time: $x^{(1)}, x^{(2)}, x^{(4)}$ use prices up to minute $k$ and σ from previous days. $x^{(3)}, x^{(6)}, x^{(7)}$ use data up to the previous close or the open. `tests/test_ml.py` checks this by changing prices after $k$.

## 10. Model and estimation

### 10.1 Standardisation

Each feature is centred and scaled with the mean and standard deviation of the **training set only**:

```math
z^{(i)}_{t,k} = \frac{x^{(i)}_{t,k} - \mu_i}{s_i}, \qquad \mu_i, s_i \text{ estimated on the training fold.}
```

This puts all coefficients on the same scale, so the L2 penalty treats them equally and their sizes are comparable.

### 10.2 Logistic regression

```math
p_{t,k} := \Pr(y_{t,k} = 1 \mid \mathbf z_{t,k}) = \frac{1}{1 + e^{-\eta_{t,k}}}, \qquad \eta_{t,k} = \beta_0 + \boldsymbol\beta^\top \mathbf z_{t,k}.
```

Equivalently, the **log-odds are linear** in the features:

```math
\ln\frac{p_{t,k}}{1-p_{t,k}} = \beta_0 + \sum_{i=1}^{7} \beta_i\, z^{(i)}_{t,k}.
```

> **Interpretation.** If feature $i$ rises by one standard deviation, the odds of an up-move are multiplied by $e^{\beta_i}$. For example, $\beta_{\text{vol regime}} = -0.148$ gives $e^{-0.148} \approx 0.86$: the odds of "up until the close" fall by about 14%. The intercept $\beta_0$ captures the base rate. In training, 52.5% of labels are 1 because of the positive drift of SPY.

### 10.3 Penalised maximum likelihood

The coefficients minimise the L2-penalised negative log-likelihood (cross-entropy). The intercept is not penalised:

```math
\hat{\boldsymbol\beta} = \arg\min_{\beta_0,\boldsymbol\beta}\;
\underbrace{-\sum_{(t,k)\in\mathcal D_{\text{train}}} \Big[ y_{t,k}\ln p_{t,k} + (1-y_{t,k})\ln(1-p_{t,k}) \Big]}_{\text{negative log-likelihood}}
\;+\; \frac{1}{2C}\,\lVert\boldsymbol\beta\rVert_2^2 .
```

- The objective is **strictly convex**, so there is a unique optimum. It is found numerically with L-BFGS (scikit-learn).
- Its gradient has a simple form, the sum of prediction errors times features plus the penalty:

```math
\nabla_{\boldsymbol\beta} = \sum_{(t,k)} \big(p_{t,k} - y_{t,k}\big)\,\mathbf z_{t,k} + \frac{1}{C}\boldsymbol\beta .
```

- **Role of $C$.** A small $C$ means a strong penalty, which shrinks the coefficients towards 0 and pushes $p$ towards the base rate (low variance, more bias). Cross-validation selected $C = 0.01$, the strongest regularisation in the grid. This is consistent with a very low signal-to-noise ratio.

### 10.4 Decision rule

With a no-trade margin $m \ge 0$:

```math
x_{t,k} =
\begin{cases}
+1 & \text{if } \hat p_{t,k} > \tfrac12 + m,\\
-1 & \text{if } \hat p_{t,k} < \tfrac12 - m,\\
\ \ 0 & \text{otherwise.}
\end{cases}
```

The position is re-evaluated at every $k \in \mathcal K_t$. A model that turns neutral therefore closes the trade, which replaces the paper's stop. Shares $q_t$, execution, costs and P&L follow §4–5 exactly.

> **Why a margin.** Every round trip pays commission and slippage twice, at entry and exit (§5). Near $\hat p = 0.5$ the expected gross gain is close to zero, so it cannot cover the costs. The margin removes these marginal trades. Its size is chosen by the backtest Sharpe after costs (§11), not by accuracy.

## 11. Model selection and evaluation

### 11.1 Expanding-window cross-validation (training period only)

For each validation year $Y \in \mathcal Y = \{2018, 2019, 2020, 2021\}$:

```math
\mathcal D^{(Y)}_{\text{fit}} = \{(t,k) \in \mathcal D : t_0 \le t < Y\}, \qquad
\mathcal D^{(Y)}_{\text{val}} = \{(t,k) \in \mathcal D : t \in Y\}.
```

The model is always fitted on the past and validated on the following year, the same way it would be used in practice. All observations of one day are always in the same fold.

For every candidate $\theta = (C, m)$ in the grid $\{0.01, 0.1, 1\} \times \{0, 0.01, 0.02, 0.04\}$, the strategy is backtested on $\mathcal D^{(Y)}_{\text{val}}$ (paper costs), and

```math
\theta^\star = \arg\max_{\theta}\; \frac{1}{|\mathcal Y|}\sum_{Y\in\mathcal Y} \mathrm{SR}_Y(\theta).
```

The feature set (A: the 7 features; B: A plus the paper's discrete signal $\in\{-1,0,1\}$) is chosen the same way. The final model is then fitted once on the whole training period 2016–2021 with $\theta^\star$ and evaluated **once** on the test period.

### 11.2 AUC

The AUC measures how well $\hat p$ *ranks* up- and down-moves, independently of any threshold. With $n_+$ positive and $n_-$ negative labels:

```math
\mathrm{AUC} = \frac{1}{n_+\,n_-} \sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0} \mathbf 1\{\hat p_i > \hat p_j\}
= \Pr\big(\hat p_{\text{up}} > \hat p_{\text{down}}\big).
```

AUC = 0.5 means no skill and 1 means perfect ranking. Our test AUC of 0.518 means that in 51.8% of all (up, down) pairs, the up-move received the higher probability.

> **Effective sample size.** The 12 observations of a day share the same close, so their labels are strongly dependent. The informative sample size is therefore closer to the number of **days** (≈1,200 in the test period) than to the number of rows (≈14,000). A naive standard error of the AUC based on all rows would be far too small. This is the same issue as the Kish effective sample size for weighted samples: correlated observations carry less information than their count suggests.

### 11.3 Combination with the paper rule (descriptive)

For two strategies with mean daily returns $\mu_i$, volatilities $\sigma_i$ and correlation $\rho$, a portfolio with weights $w_1 + w_2 = 1$ has

```math
\mathrm{SR}_{\text{comb}} = \frac{w_1\mu_1 + w_2\mu_2}{\sqrt{w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\,\sigma_1\sigma_2}}\;\sqrt{252}.
```

For $\rho \approx 0$ the denominator shrinks, which raises the combined Sharpe ratio. With inverse-volatility weights $w_i \propto 1/\sigma_i$ (ML 0.39, rule 0.61) and the measured $\rho = -0.03$, the test Sharpe of the mix is about 1.15. The weights were **not** chosen on the training period, so this is a hypothesis for further work, not a result.

## 12. Parameters of the ML strategy

| Symbol | Meaning | Value |
|---|---|---|
| $\mathcal K_t$ | decision times | every 30 min, 10:00–15:30 (as in Part I) |
| $\mathbf x$ | features | 7 (feature set A, chosen by CV) |
| $C$ | inverse L2 strength | 0.01 (CV) |
| $m$ | no-trade margin | 0.02 (CV) |
| $\mathcal Y$ | validation years | 2018–2021 |
| sizing | 1x or volatility targeting | as in §4 |
