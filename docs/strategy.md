# Intraday Momentum on SPY – Mathematical Specification

*Based on Zarattini, Aziz & Barbon (2024), "Beat the Market". This document states every step of the strategy and its evaluation as a formula, in the same order as the code in [`src/intraday_momentum/`](../src/intraday_momentum). Each section names the module that implements it.*

**Contents**

1. [Notation and information structure](#1-notation-and-information-structure)
2. [The Noise Area](#2-the-noise-area-signalspy)
3. [Signal and position](#3-signal-and-position-backtestpy)
4. [Position sizing](#4-position-sizing)
5. [Profit and loss with transaction costs](#5-profit-and-loss-with-transaction-costs)
6. [Performance metrics](#6-performance-metrics-metricspy)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Summary of parameters](#8-summary-of-parameters)

---

## 1. Notation and information structure

**Days and minutes.** Trading days are indexed by $t = 1,\dots,T$. The regular session (09:30–16:00 ET) consists of $N = 390$ one-minute bars $j = 0,\dots,N-1$. Bar $j$ covers the interval $[09{:}30 + j,\ 09{:}30 + j + 1)$. On half days the session ends earlier. In general $N_t \le N$ denotes the number of bars on day $t$ (`data.py`).

**Prices.** Each bar has open, high, low, close and volume $(O_{t,j}, H_{t,j}, L_{t,j}, C_{t,j}, V_{t,j})$. The price observed $k$ minutes after the open is the close of the bar that ends at $09{:}30+k$:

```math
P_{t,k} := C_{t,k-1}, \qquad k = 1,\dots,N_t .
```

The open and close of the day are

```math
O_t := O_{t,0}, \qquad C_t := P_{t,N_t}.
```

**Information.** $\mathcal F_{t,k}$ denotes the information available at minute $k$ of day $t$: all bars of days $1,\dots,t-1$ and bars $0,\dots,k-1$ of day $t$.

> **No look-ahead bias** ⇔ every decision taken at $(t,k)$ is $\mathcal F_{t,k}$-measurable *and* is executed at a price realised **after** $(t,k)$.

Every quantity below is marked with the information set it depends on.

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

The sum starts at $i=1$, so $\sigma_{t,k}$ uses only previous days and is $\mathcal F_{t,0}$-measurable. In the code this is a rolling mean over days, shifted by one day.

> **Remark – why the band widens like $\sqrt{k}$.** $\sigma_{t,k}$ is a mean *absolute* move, not a standard deviation. Suppose one-minute log returns are i.i.d. $\mathcal N(0,s^2)$. Then $\ln(P_{t,k}/O_t) \sim \mathcal N(0, k s^2)$, and for small moves
>
> ```math
> \mathbb E[m_{t,k}] \approx \mathbb E|Z|\, s\sqrt{k} = \sqrt{\tfrac{2}{\pi}}\, s\sqrt{k}, \qquad Z\sim\mathcal N(0,1).
> ```
>
> The threshold for a "significant" move therefore grows like $\sqrt{k}$. This gives the funnel-shaped Noise Area in Figure 1 of the paper. A fixed threshold would be too loose in the morning and too tight in the afternoon.

### 2.2 Boundaries with gap adjustment

Let $C_{t-1}$ be the previous close and $D_t$ the cash dividend if $t$ is an ex-dividend date ($D_t = 0$ otherwise). The dividend-adjusted previous close is $\tilde C_{t-1} = C_{t-1} - D_t$. With the volatility multiplier $\mathrm{VM}$ (paper: $\mathrm{VM}=1$):

```math
\begin{aligned}
\mathrm{UB}_{t,k} &= \max\!\big(O_t,\ \tilde C_{t-1}\big)\,\big(1 + \mathrm{VM}\,\sigma_{t,k}\big),\\
\mathrm{LB}_{t,k} &= \min\!\big(O_t,\ \tilde C_{t-1}\big)\,\big(1 - \mathrm{VM}\,\sigma_{t,k}\big).
\end{aligned}
```

The Noise Area is the interval $[\mathrm{LB}_{t,k},\ \mathrm{UB}_{t,k}]$.

> **Remark – gap adjustment.** After a gap down ($O_t < \tilde C_{t-1}$), the upper band is anchored at yesterday's close. A long signal then requires the price to close the whole gap **and** to move a further $\sigma_{t,k}$. The lower band stays anchored at the open, so a continuation of the gap is detected quickly. Gap ups work the same way in the other direction. Without the dividend correction, the mechanical price drop on ex-dates would look like a gap down.

### 2.3 Session VWAP

With the typical price $\bar p_{t,j} = (H_{t,j}+L_{t,j}+C_{t,j})/3$:

```math
\mathrm{VWAP}_{t,k} = \frac{\sum_{j=0}^{k-1} \bar p_{t,j}\,V_{t,j}}{\sum_{j=0}^{k-1} V_{t,j}} .
```

It uses only bars up to minute $k$, so it is $\mathcal F_{t,k}$-measurable.

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
\hat P_{t,k} = O_{t,k} \qquad (\text{realised after } \mathcal F_{t,k}).
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

which is $\mathcal F_{t,0}$-measurable. Then

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
\kappa(q) = \max\big(c_{\min},\ c\,q\big) + \delta\, q ,
```

with commission $c$ = \$0.0035 per share, minimum $c_{\min}$ = \$0.35 per order and slippage $\delta$ per share (\$0.001 in the paper scenario, \$0.005 in the conservative scenario).

**Equity and returns:**

```math
A_t = A_{t-1} + \Pi_t, \qquad R^{\text{strat}}_t = \frac{\Pi_t}{A_{t-1}}, \qquad A_0 = \$100{,}000 .
```

> **Proposition – break-even cost.** Let $\bar\pi$ be the average gross profit per share per round trip. A round trip consists of two orders. For large $q$, where the minimum commission does not bind, the strategy is profitable on average if and only if
>
> ```math
> \bar\pi > 2\,(c + \delta).
> ```
>
> *Proof.* On average a round trip earns $q\,\bar\pi$ and costs $2\kappa(q) = 2q(c+\delta)$ when $cq \ge c_{\min}$. ∎
>
> The paper reports $\bar\pi \approx$ \$0.09, so the edge disappears at roughly $c+\delta \approx$ \$0.045 per share and side. This is why cost assumptions matter so much for an intraday strategy.

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
- **No look-ahead.** $\sigma_{t,k}$, $\hat\sigma_t$, $\tilde C_{t-1}$ and $L_t$ are $\mathcal F_{t,0}$-measurable. $P_{t,k}$ and $\mathrm{VWAP}_{t,k}$ are $\mathcal F_{t,k}$-measurable. Execution happens at $O_{t,k}$, after $\mathcal F_{t,k}$. Unit tests check these properties (`tests/test_signals.py`).
- **Costs.** Two scenarios ($\delta = 0.001$ and $\delta = 0.005$), plus a sweep over total costs $c+\delta \in [0.0035,\ 0.0235]$ per share in the test period.

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
| $\delta$ | slippage per share | \$0.001 / \$0.005 |
| $A_0$ | initial capital | \$100,000 |
