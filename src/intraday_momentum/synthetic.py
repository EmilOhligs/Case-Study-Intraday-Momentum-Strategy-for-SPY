"""Synthetic minute bars for tests and smoke runs (no market data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TZ


def make_minute_bars(n_days: int = 60, start: str = "2024-01-02", seed: int = 0,
                     daily_vol: float = 0.01, drift_prob: float = 0.3, s0: float = 400.0) -> pd.DataFrame:
    """Random-walk SPY-like minute bars. On a fraction `drift_prob` of days a trend is added."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    minute_vol = daily_vol / np.sqrt(390)
    frames, price = [], s0
    for d in days:
        price *= np.exp(rng.normal(0, 0.003))                         # overnight gap
        trend = rng.choice([-1, 1]) * daily_vol / 390 * 2 if rng.random() < drift_prob else 0.0
        log_ret = rng.normal(trend, minute_vol, 390)
        closes = price * np.exp(np.cumsum(log_ret))
        opens = np.concatenate([[price], closes[:-1]])
        noise = np.abs(rng.normal(0, minute_vol / 2, 390)) * closes
        idx = pd.date_range(pd.Timestamp(d).tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30), periods=390, freq="min")
        frames.append(pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + noise,
                                    "low": np.minimum(opens, closes) - noise, "close": closes,
                                    "volume": rng.integers(50_000, 150_000, 390).astype(float)}, index=idx))
        price = closes[-1]
    return pd.concat(frames)
