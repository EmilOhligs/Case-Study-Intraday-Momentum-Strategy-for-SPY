"""Trade-level diagnostics: where does the P&L come from, and how often are breakouts false?"""
from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult


def _trades(result: BacktestResult, start=None, end=None) -> pd.DataFrame:
    t = result.trades.copy()
    if t.empty:
        return t
    if start is not None:
        t = t[t["date"] >= pd.Timestamp(start)]
    if end is not None:
        t = t[t["date"] <= pd.Timestamp(end)]
    t["pnl_per_share"] = t["gross_pnl"] / t["shares"]
    t["ret_bps"] = 1e4 * t["side"] * (t["exit_px"] / t["entry_px"] - 1)
    # column j = bar starting at 09:30 + j -> decision time 09:30 + j + 1
    t["entry_time"] = (pd.Timestamp("09:30") + pd.to_timedelta(t["entry_col"] + 1, unit="min")).dt.strftime("%H:%M")
    return t


def false_breakout_stats(result: BacktestResult, start=None, end=None, decision_every_min: int = 30) -> pd.Series:
    """Share of entries that are closed again at the very next decision time (not at the close).

    These are the "false breakouts" Idea 1 (transient-spike filter) targets.
    """
    t = _trades(result, start, end)
    if t.empty:
        return pd.Series(dtype=float)
    false = (t["exit_reason"] == "signal") & (t["exit_col"] - t["entry_col"] == decision_every_min)
    return pd.Series({
        "n_trades": len(t),
        "false_breakout_rate": false.mean(),
        "avg_bps_false": t.loc[false, "ret_bps"].mean(),
        "avg_bps_other": t.loc[~false, "ret_bps"].mean(),
    })


def pnl_by_entry_time(result: BacktestResult, start=None, end=None) -> pd.DataFrame:
    """Number of trades, hit ratio and average gross return (bp) by entry time (cf. paper FAQ Q18)."""
    t = _trades(result, start, end)
    return t.groupby("entry_time").agg(n_trades=("ret_bps", "size"), hit_ratio=("ret_bps", lambda x: (x > 0).mean()),
                                       avg_bps=("ret_bps", "mean"))


def pnl_by_side(result: BacktestResult, start=None, end=None) -> pd.DataFrame:
    """Long vs short contribution (cf. paper FAQ Q5)."""
    t = _trades(result, start, end)
    t["side"] = t["side"].map({1: "long", -1: "short"})
    return t.groupby("side").agg(n_trades=("ret_bps", "size"), hit_ratio=("ret_bps", lambda x: (x > 0).mean()),
                                 avg_bps=("ret_bps", "mean"), total_gross_pnl=("gross_pnl", "sum"))


def yearly_returns(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Calendar-year compounded returns per strategy (cf. paper monthly table, FAQ Q4)."""
    return pd.DataFrame({name: (1 + r.fillna(0)).groupby(r.index.year).prod() - 1 for name, r in returns.items()})
