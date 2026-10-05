"""Signal construction: the Noise Area of Zarattini, Aziz & Barbon (2024), Section 3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DayData


def average_move_from_open(data: DayData, lookback_days: int) -> np.ndarray:
    """sigma[t, j] = mean over the previous `lookback_days` days of |close[t-i, j] / open[t-i] - 1|.

    Uses only days t-1 ... t-lookback (shift by one day), so there is no look-ahead.
    Half days (NaN after the early close) are skipped as long as 80% of the window is available.
    """
    move = np.abs(data.close / data.day_open[:, None] - 1.0)
    min_periods = int(np.ceil(0.8 * lookback_days))
    sigma = pd.DataFrame(move).rolling(lookback_days, min_periods=min_periods).mean().shift(1)
    return sigma.values


def noise_area(data: DayData, lookback_days: int = 14, vol_multiplier: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower boundary of the Noise Area, including the overnight-gap adjustment.

    UB = max(Open_t, Close_{t-1}) * (1 + VM * sigma)
    LB = min(Open_t, Close_{t-1}) * (1 - VM * sigma)
    """
    sigma = average_move_from_open(data, lookback_days)
    upper_anchor = np.fmax(data.day_open, data.prev_close)[:, None]   # fmax ignores a missing prev_close
    lower_anchor = np.fmin(data.day_open, data.prev_close)[:, None]
    return upper_anchor * (1 + vol_multiplier * sigma), lower_anchor * (1 - vol_multiplier * sigma)


def trailing_daily_vol(data: DayData, lookback_days: int = 14) -> np.ndarray:
    """Std of the last `lookback_days` daily close-to-close returns, known before day t opens."""
    return data.daily_returns.rolling(lookback_days).std(ddof=1).shift(1).values
