"""Features and labels for the ML strategy (one row per day x decision time).

Every feature is computed from information available at the decision time (no look-ahead):
prices up to the decision minute, the paper's Noise-Area sigma (previous 14 days only) and
daily statistics up to the previous close. The label uses the future and is only used as target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DayData
from .signals import average_move_from_open, noise_area, trailing_daily_vol

FEATURES = ["move_sigma", "vwap_sigma", "gap_z", "ret30_z", "time_of_day", "rsi5", "vol_regime"]
# Variant B adds the paper's own discrete signal, so the model nests the rule-based strategy
FEATURES_WITH_PAPER_SIGNAL = FEATURES + ["paper_signal"]

FEATURE_DESCRIPTIONS = {
    "move_sigma": "move from the open in units of the Noise-Area sigma: (P/O - 1) / sigma  (the paper's signal, continuous)",
    "vwap_sigma": "distance to VWAP in sigma units: (P/VWAP - 1) / sigma",
    "gap_z": "overnight gap (dividend-adjusted) / trailing 14-day daily vol",
    "ret30_z": "return over the last 30 minutes / trailing 14-day daily vol",
    "time_of_day": "minutes since the open / 390",
    "rsi5": "5-day RSI of SPY at the previous close, rescaled to [-1, 1] (dealer-gamma proxy, paper 4.5)",
    "vol_regime": "log(5-day vol / 60-day vol) of daily returns (volatility regime, paper 4.1)",
    "paper_signal": "+1 if P > max(UB, VWAP), -1 if P < min(LB, VWAP), else 0 (the paper's Ext. 1 rule)",
}


def rsi(close: pd.Series, n: int = 5) -> pd.Series:
    """Wilder's RSI on a price series (value at t uses closes up to t)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def decision_columns(first: int = 30, every: int = 30) -> np.ndarray:
    """Matrix columns of the decision times 10:00, 10:30, ..., 15:30 (column j = time 09:30 + j + 1)."""
    return np.arange(first - 1, 390, every)


def build_features(data: DayData, lookback_days: int = 14, first_decision_min: int = 30,
                   decision_every_min: int = 30) -> pd.DataFrame:
    """Long table: index (date, col); columns = FEATURES + ['fwd_ret', 'label'].

    fwd_ret = return from the execution price (open of the next bar) to the day's close,
    label   = 1 if fwd_ret > 0 else 0.
    Rows with missing features (burn-in) or at/after the close are dropped.
    """
    cols = decision_columns(first_decision_min, decision_every_min)
    sigma = average_move_from_open(data, lookback_days)
    vol14 = trailing_daily_vol(data, 14)
    vol5, vol60 = trailing_daily_vol(data, 5), trailing_daily_vol(data, 60)
    rsi_prev = rsi(pd.Series(data.day_close, index=data.dates), 5).shift(1).values   # known before the open
    gap = data.day_open / data.prev_close - 1

    n = len(data)
    price = data.close[:, cols]
    price_30_ago = np.where(cols >= 30, data.close[:, np.maximum(cols - 30, 0)], data.day_open[:, None])
    frame = {
        "move_sigma": (price / data.day_open[:, None] - 1) / sigma[:, cols],
        "vwap_sigma": (price / data.vwap[:, cols] - 1) / sigma[:, cols],
        "gap_z": np.repeat((gap / vol14)[:, None], len(cols), axis=1),
        "ret30_z": (price / price_30_ago - 1) / vol14[:, None],
        "time_of_day": np.repeat(((cols + 1) / 390)[None, :], n, axis=0),
        "rsi5": np.repeat(((rsi_prev - 50) / 50)[:, None], len(cols), axis=1),
        "vol_regime": np.repeat(np.log(vol5 / vol60)[:, None], len(cols), axis=1),
    }
    ub, lb = noise_area(data, lookback_days, 1.0)
    vw = data.vwap[:, cols]
    frame["paper_signal"] = np.where(price > np.fmax(ub[:, cols], vw), 1.0,
                                     np.where(price < np.fmin(lb[:, cols], vw), -1.0, 0.0))
    exec_px = data.nxt_open[:, cols]
    fwd_ret = data.day_close[:, None] / exec_px - 1
    before_close = cols[None, :] < (data.n_bars[:, None] - 1)

    idx = pd.MultiIndex.from_product([data.dates, cols], names=["date", "col"])
    df = pd.DataFrame({k: v.ravel() for k, v in frame.items()}, index=idx)
    df["fwd_ret"] = fwd_ret.ravel()
    df["label"] = (df["fwd_ret"] > 0).astype(int)
    df = df[before_close.ravel()]
    return df.replace([np.inf, -np.inf], np.nan).dropna()
