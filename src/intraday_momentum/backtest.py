"""Event-driven daily loop. Decisions only at fixed times (HH:00 / HH:30), always flat at the close.

Accounting per day t:
    shares_t  = floor(AUM_{t-1} * leverage_t / Open_t)         (fixed for the whole day)
    PnL_t     = sum over trades of side * shares * (exit - entry) - n_orders * order_cost(shares)
    AUM_t     = AUM_{t-1} + PnL_t
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostConfig, StrategyConfig
from .data import DayData
from .signals import noise_area, trailing_daily_vol


@dataclass
class BacktestResult:
    config: StrategyConfig
    costs: CostConfig
    daily: pd.DataFrame          # index = date; columns: ret, aum, gross_pnl, costs, leverage, n_trades
    trades: pd.DataFrame         # one row per round trip

    @property
    def returns(self) -> pd.Series:
        return self.daily["ret"]


@dataclass
class _Signals:
    upper: np.ndarray
    lower: np.ndarray
    leverage: np.ndarray


def _target_position(cfg: StrategyConfig, pos: int, price: float, ub: float, lb: float, vwap: float) -> int:
    """New position after observing `price` at a decision time."""
    if cfg.stop == "opposite_band":
        # Enter on a breakout; an open position is only closed / reversed by the opposite band.
        if price > ub:
            return 1
        if price < lb:
            return -1
        return pos
    if cfg.stop == "band_vwap":
        # Long only while price is above BOTH the upper band and VWAP (= trailing stop max(UB, VWAP)).
        if price > max(ub, vwap):
            return 1
        if price < min(lb, vwap):
            return -1
        return 0
    raise ValueError(f"unknown stop rule {cfg.stop!r}")


def compute_signals(data: DayData, cfg: StrategyConfig) -> _Signals:
    upper, lower = noise_area(data, cfg.lookback_days, cfg.vol_multiplier)
    if cfg.sizing == "full":
        leverage = np.ones(len(data))
    elif cfg.sizing == "vol_target":
        vol = trailing_daily_vol(data, cfg.vol_lookback_days)
        with np.errstate(divide="ignore", invalid="ignore"):
            leverage = np.minimum(cfg.max_leverage, cfg.vol_target_daily / vol)
    else:
        raise ValueError(f"unknown sizing {cfg.sizing!r}")
    return _Signals(upper=upper, lower=lower, leverage=leverage)


def run_backtest(data: DayData, cfg: StrategyConfig = StrategyConfig(), costs: CostConfig = CostConfig(),
                 start: str | None = None, end: str | None = None, initial_capital: float = 100_000.0,
                 day_filter: np.ndarray | None = None,
                 target_positions: np.ndarray | None = None) -> BacktestResult:
    """Simulate the strategy on days in [start, end].

    Signals are computed on the full history (so the first test day already has a 14-day lookback),
    but P&L starts at `start` with `initial_capital`.
    `day_filter` (optional, bool per day) lets a variant skip days entirely - it must be known before the open.
    `target_positions` (optional, days x 390, values in {-1, 0, +1}, NaN = no view) replaces the paper's
    signal rule: at every decision time the position is set to this value. Used by the ML strategy.
    Execution, sizing (`cfg.sizing`), costs and the flat-at-close rule are identical for all strategies.
    """
    sig = compute_signals(data, cfg)
    in_range = np.ones(len(data), dtype=bool)
    if start is not None:
        in_range &= data.dates >= pd.Timestamp(start)
    if end is not None:
        in_range &= data.dates <= pd.Timestamp(end)

    decision_cols = np.arange(cfg.first_decision_min - 1, 390, cfg.decision_every_min)
    aum = initial_capital
    daily_rows, trades = [], []

    for i in np.flatnonzero(in_range):
        date = data.dates[i]
        last = data.n_bars[i] - 1                       # column of the closing bar
        lev = sig.leverage[i]
        tradable = (np.isfinite(sig.upper[i, decision_cols[0]]) and np.isfinite(lev)
                    and (day_filter is None or day_filter[i]))
        shares = int(np.floor(aum * lev / data.day_open[i])) if tradable else 0

        pos, entry_px, entry_col = 0, np.nan, -1
        gross, n_orders, n_round_trips = 0.0, 0, 0

        def close_position(exit_px: float, exit_col: int, reason: str) -> None:
            nonlocal gross, n_orders, n_round_trips
            pnl = pos * shares * (exit_px - entry_px)
            gross += pnl
            n_orders += 1
            n_round_trips += 1
            trades.append({"date": date, "side": pos, "entry_col": entry_col, "exit_col": exit_col,
                           "entry_px": entry_px, "exit_px": exit_px, "shares": shares, "gross_pnl": pnl,
                           "exit_reason": reason})

        if shares > 0:
            for j in decision_cols[decision_cols < last]:
                price = data.close[i, j]
                if not np.isfinite(price):
                    continue
                if target_positions is not None:
                    tp = target_positions[i, j]
                    new_pos = 0 if np.isnan(tp) else int(tp)
                else:
                    new_pos = _target_position(cfg, pos, price, sig.upper[i, j], sig.lower[i, j], data.vwap[i, j])
                if new_pos == pos:
                    continue
                exec_px = data.nxt_open[i, j]             # executed at the open of the next bar
                if pos != 0:
                    close_position(exec_px, j, "signal")
                if new_pos != 0:
                    entry_px, entry_col = exec_px, j
                    n_orders += 1
                pos = new_pos
            if pos != 0:                                  # flat at the close
                close_position(data.close[i, last], last, "close")
                pos = 0

        cost = n_orders * costs.order_cost(shares) if n_orders else 0.0
        pnl = gross - cost
        ret = pnl / aum
        aum += pnl
        daily_rows.append({"date": date, "ret": ret, "aum": aum, "gross_pnl": gross, "costs": cost,
                           "leverage": lev if shares > 0 else 0.0, "n_trades": n_round_trips})

    daily = pd.DataFrame(daily_rows).set_index("date")
    trades_df = pd.DataFrame(trades)
    return BacktestResult(config=cfg, costs=costs, daily=daily, trades=trades_df)
