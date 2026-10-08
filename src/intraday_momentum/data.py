"""Loading minute bars and reshaping them into (days x minutes) matrices.

Time convention (important for avoiding look-ahead bias)
--------------------------------------------------------
A bar labelled 09:30 covers [09:30, 09:31). Its close is the price *at* 09:31.
Column j of the matrices therefore refers to the bar starting at 09:30 + j minutes:

    close[:, j]  = price observed at time 09:30 + (j + 1) min    (known at that time)
    vwap[:, j]   = session VWAP using bars 0..j                  (known at that time)
    nxt_open[:, j] = open of bar j + 1  -> the earliest realistic execution price
                     for a decision taken at time 09:30 + (j + 1) min

So the decision at 10:00 looks at column 29 and trades at the open of the 10:00 bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
MINUTES_PER_SESSION = 390  # 09:30 - 16:00
EARLY_CLOSE_MINUTES = 210  # 09:30 - 13:00 on NYSE half days
MIN_DIVIDEND = 0.05        # smaller "dividends" are rounding noise from adjusted vs. raw prices


def nyse_early_closes(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """NYSE half days (13:00 close) among the given trading days.

    Rules: day after Thanksgiving; Dec 24 and Jul 3 if they are trading days on Mon-Thu
    (if they fall on a Friday the exchange is closed for the observed holiday instead).
    Without this, the vendor's after-hours bars between 13:00 and 16:00 would be treated
    as regular-session bars.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    is_thanksgiving_friday = (dates.month == 11) & (dates.dayofweek == 4) & (dates.day >= 23) & (dates.day <= 29)
    is_eve = (((dates.month == 12) & (dates.day == 24)) | ((dates.month == 7) & (dates.day == 3))) & (dates.dayofweek <= 3)
    return dates[is_thanksgiving_friday | is_eve]


@dataclass
class DayData:
    dates: pd.DatetimeIndex        # one entry per trading day
    day_open: np.ndarray           # (n,)  open of the 09:30 bar
    prev_close: np.ndarray         # (n,)  last regular-session close of the previous day
    day_close: np.ndarray          # (n,)  last regular-session close of the day
    n_bars: np.ndarray             # (n,)  number of regular-session minutes (390, or 210 on half days)
    close: np.ndarray              # (n, 390)
    nxt_open: np.ndarray           # (n, 390)
    vwap: np.ndarray               # (n, 390)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def daily_returns(self) -> pd.Series:
        """Close-to-close returns of SPY (unadjusted). Known at the end of each day."""
        return pd.Series(self.day_close / self.prev_close - 1.0, index=self.dates)


def load_minute_bars(path: str | Path) -> pd.DataFrame:
    """Load one parquet/csv file or a directory of them; return bars indexed by New York time."""
    path = Path(path)
    files = sorted(path.glob("*.parquet")) + sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet/csv files in {path}. Run scripts/download_data.py first.")
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, index_col=0) for f in files]
    bars = pd.concat(frames)
    idx = pd.to_datetime(bars.index, utc=True)
    bars.index = idx.tz_convert(TZ)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    return bars[["open", "high", "low", "close", "volume"]].astype(float)


def _regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    minutes = bars.index.hour * 60 + bars.index.minute - (9 * 60 + 30)
    rth = bars[(minutes >= 0) & (minutes < MINUTES_PER_SESSION)].copy()
    rth["date"] = rth.index.normalize().tz_localize(None)
    rth["minute"] = minutes[(minutes >= 0) & (minutes < MINUTES_PER_SESSION)]
    half_days = nyse_early_closes(pd.DatetimeIndex(rth["date"].unique()))
    after_early_close = rth["date"].isin(half_days) & (rth["minute"] >= EARLY_CLOSE_MINUTES)
    return rth[~after_early_close]


def build_day_data(bars: pd.DataFrame, min_bars: int = 180, dividends: pd.Series | None = None) -> DayData:
    """Pivot regular-session minute bars into day x minute matrices.

    `dividends` (cash amount indexed by ex-date) is subtracted from the previous close on the ex-date,
    so the mechanical price drop on ex-dividend days is not mistaken for an overnight gap
    (same treatment as the authors' reference code).

    Missing minutes inside a session are forward-filled (no trade in that minute = price unchanged);
    minutes after an early close stay NaN. Days with fewer than `min_bars` bars are dropped.
    """
    rth = _regular_session(bars)
    cols = np.arange(MINUTES_PER_SESSION)
    pivot = lambda field: rth.pivot(index="date", columns="minute", values=field).reindex(columns=cols)  # noqa: E731

    close_raw = pivot("close")
    last_bar = close_raw.notna().values[:, ::-1].argmax(axis=1)          # distance of last valid bar from the end
    n_bars = MINUTES_PER_SESSION - last_bar
    keep = (close_raw.notna().sum(axis=1).values >= min_bars)

    after_close = cols[None, :] >= n_bars[:, None]
    close = close_raw.ffill(axis=1).to_numpy(copy=True)   # writable copy (pandas >= 3 returns read-only views)
    close[after_close] = np.nan

    opens = pivot("open").values
    day_open = np.where(np.isnan(opens[:, 0]), close_raw.bfill(axis=1).values[:, 0], opens[:, 0])
    nxt_open = np.full_like(close, np.nan)
    nxt_open[:, :-1] = opens[:, 1:]
    nxt_open = np.where(np.isnan(nxt_open), close, nxt_open)   # fallback: last known price

    high, low = pivot("high").values, pivot("low").values
    vol = np.nan_to_num(pivot("volume").values)
    typical = np.nan_to_num((high + low + close_raw.values) / 3.0)
    cum_vol = np.cumsum(vol, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(typical * vol, axis=1) / cum_vol
    vwap = pd.DataFrame(np.where(cum_vol > 0, vwap, np.nan)).ffill(axis=1).to_numpy(copy=True)
    vwap[after_close] = np.nan

    day_close = close[np.arange(len(close)), n_bars - 1]
    dates = pd.DatetimeIndex(close_raw.index)

    # previous session's close, taken before short days are dropped, so a dropped day does not
    # shift the close of two days ago into the gap adjustment
    prev_close = np.concatenate([[np.nan], day_close[:-1]])
    if dividends is not None:
        prev_close = prev_close - dividends.reindex(dates).fillna(0.0).values

    sel = keep
    day_close, day_open, n_bars, prev_close = day_close[sel], day_open[sel], n_bars[sel], prev_close[sel]
    return DayData(dates=dates[sel], day_open=day_open, prev_close=prev_close, day_close=day_close,
                   n_bars=n_bars, close=close[sel], nxt_open=nxt_open[sel], vwap=vwap[sel])


def load_dividends(path: str | Path) -> pd.Series:
    """Cash dividends per share indexed by ex-date (written by scripts/download_data.py)."""
    div = pd.read_csv(path, index_col=0, parse_dates=True)["dividend"]
    return div[div >= MIN_DIVIDEND]


def load_daily_benchmark(path: str | Path) -> pd.Series:
    """Daily total-return series of SPY from adjusted daily bars (dividends included)."""
    daily = pd.read_parquet(path)
    idx = pd.to_datetime(daily.index, utc=True).tz_convert(TZ).normalize().tz_localize(None)
    return pd.Series(daily["close"].values, index=idx).pct_change()
