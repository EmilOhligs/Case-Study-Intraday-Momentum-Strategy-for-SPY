"""Performance metrics. All inputs are daily simple returns; 252 trading days per year."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def annualized_return(r: pd.Series) -> float:
    """Geometric (CAGR): (1 + total)^(252 / N) - 1."""
    return float((1 + total_return(r)) ** (TRADING_DAYS / len(r)) - 1)


def annualized_volatility(r: pd.Series) -> float:
    """Daily std scaled by sqrt(252): variances of independent returns add up linearly in time."""
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe = mean(excess daily return) / std(daily return) * sqrt(252)."""
    excess = r - rf_annual / TRADING_DAYS
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough loss of the compounded equity curve (positive number)."""
    equity = (1 + r).cumprod()
    return float(-(equity / equity.cummax() - 1).min())


def summarize(r: pd.Series, trades: pd.DataFrame | None = None, rf_annual: float = 0.0) -> dict[str, float]:
    r = r.dropna()
    traded = r[r != 0]
    out = {
        "start": r.index[0].date(), "end": r.index[-1].date(), "days": len(r),
        "total_return": total_return(r),
        "ann_return": annualized_return(r),
        "ann_vol": annualized_volatility(r),
        "sharpe": sharpe_ratio(r, rf_annual),
        "sharpe_t": sharpe_ratio(r, rf_annual) * np.sqrt(len(r) / TRADING_DAYS),  # approx. t-stat of mean
        "max_drawdown": max_drawdown(r),
        "hit_ratio": float((traded > 0).mean()) if len(traded) else np.nan,
        "skew": float(r.skew()),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
    }
    if trades is not None and len(trades):
        t = trades[(trades["date"] >= r.index[0]) & (trades["date"] <= r.index[-1])]
        out["n_trades"] = len(t)
        out["trades_per_day"] = len(t) / len(r)
        out["avg_gross_pnl_per_share"] = float((t["gross_pnl"] / t["shares"]).mean()) if len(t) else np.nan
    return out
