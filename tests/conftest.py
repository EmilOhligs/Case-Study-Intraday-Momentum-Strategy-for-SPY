import numpy as np
import pandas as pd
import pytest

from intraday_momentum.data import TZ


def bars_from_paths(paths: list[np.ndarray], start: str = "2024-01-02") -> pd.DataFrame:
    """Turn a list of per-day price paths (length 391: open + 390 closes) into minute bars.

    Each bar's open equals the previous bar's close, so execution prices are easy to predict.
    """
    frames = []
    for d, path in zip(pd.bdate_range(start, periods=len(paths)), paths):
        opens, closes = path[:-1], path[1:]
        idx = pd.date_range(pd.Timestamp(d).tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30), periods=390, freq="min")
        frames.append(pd.DataFrame({"open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes),
                                    "close": closes, "volume": 1000.0}, index=idx))
    return pd.concat(frames)


def quiet_day(level: float = 100.0, amp: float = 0.0002) -> np.ndarray:
    """Price wiggles by +-amp around `level` and ends where it started (no trend)."""
    return level * (1 + amp * np.sin(np.linspace(0, 6 * np.pi, 391)))


@pytest.fixture
def history() -> list[np.ndarray]:
    return [quiet_day() for _ in range(20)]
