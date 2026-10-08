# Case Study: Intraday Momentum Strategy for SPY (WUTIS Superday)

*Original assignment, transcribed from the case-study PDF provided by WUTIS.*

---

Dear applicant,

Congratulations again on being invited to the Superday. With this case study, we aim to test the two most important skills for the Algorithmic Trading department: the ability to read and understand academic literature and the ability to write clear, reliable code.

Please use the attached paper *Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)* as your primary source.

## The tasks are

### 1. Read and understand the paper thoroughly.
- Be ready to discuss the core hypothesis, signal construction, execution logic, and risk controls.
- Be ready to explain why the strategy may work and in which regimes it may fail.

### 2. Implement and backtest the strategy (or a well-justified core subset).
- If the full paper implementation is too complex, implement a robust base version first, then add one extension.
- At minimum, include a train/test split and realistic transaction costs/slippage assumptions.
- Plot and compare **Sharpe ratio**, **annualized return**, and **annualized volatility**.
- Suggested references: Investopedia: Sharpe Ratio, Investopedia: Annualized Return, Investopedia: Volatility.

### 3. Invent your own strategy inspired by the paper and evaluate it.
- Propose one original variation (examples: ML-based model that predicts long/short/flat intraday exposure; different adjustment of the thresholds or redefinition of the stopping logic; usage of additional financial data, etc.).
- Compare it against your baseline using the same data split, cost assumptions, and evaluation metrics.
- Explain your design choices, expected edge, and further improvements.

## A few remarks regarding evaluation
- "It is better to be roughly right than precisely wrong."
- Show us that you spent meaningful time with the task, even if some parts are incomplete.
- Readability and structure of code matter.
- We will consider your current study progress and prior coding experience.

Good luck with the task, we look forward to discussing your case study with you.

*Reference paper: Zarattini, Aziz, Barbon (2025), Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY).*

---

## Where each task is covered in this repository

| Task | Covered in |
|---|---|
| 1. Paper: hypothesis, signal, execution, risk controls, regimes | [`results.md`](results.md) §1, [`strategy.md`](strategy.md) Part I |
| 2. Implementation, train/test split, costs, Sharpe / return / volatility | `src/intraday_momentum/`, [`../notebooks/backtest.ipynb`](../notebooks/backtest.ipynb) §4–7, [`results.md`](results.md) §3–5 |
| 3. Own strategy (ML long/short/flat model) | [`extensions.md`](extensions.md), [`strategy.md`](strategy.md) Part II, notebook §8, [`results.md`](results.md) §6–7 |
