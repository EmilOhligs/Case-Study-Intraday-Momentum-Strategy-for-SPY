"""Reusable evaluation routines, shared by scripts/run_backtest.py and the notebook."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import CostConfig, StrategyConfig
from .data import DayData, build_day_data, load_daily_benchmark, load_dividends, load_minute_bars
from .metrics import summarize
from .synthetic import make_minute_bars

# Strategy variants of the paper (all other parameters = paper defaults)
PAPER_VARIANTS: dict[str, StrategyConfig] = {
    "Base: opposite-band stop": StrategyConfig(name="base", stop="opposite_band", sizing="full"),
    "Ext. 1: + band/VWAP stop": StrategyConfig(name="vwap", stop="band_vwap", sizing="full"),
    "Ext. 2: + vol targeting": StrategyConfig(name="paper", stop="band_vwap", sizing="vol_target"),
}

# Cost scenarios
COST_SCENARIOS: dict[str, CostConfig] = {
    "paper": CostConfig(commission_per_share=0.0035, slippage_per_share=0.001),
    "conservative": CostConfig(commission_per_share=0.0035, slippage_per_share=0.005),  # ~half the SPY spread
}

BURN_IN_DAYS = 20   # >= lookback (14) + margin, so the first evaluated day has a full Noise Area
MAIN_COLUMNS = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_ratio", "skew",
                "alpha_ann", "beta", "trades_per_day"]
PCT_COLUMNS = ["total_return", "ann_return", "ann_vol", "max_drawdown", "hit_ratio", "worst_day",
               "best_day", "alpha_ann"]


def load_project_data(root: str | Path, synthetic: bool = False) -> tuple[DayData, pd.Series]:
    """Load minute bars (+ dividends) and the buy & hold benchmark from <root>/data.

    With `synthetic=True` (or if no data has been downloaded yet) random-walk data is used instead,
    so the full pipeline can be run without an API key.
    """
    root = Path(root)
    raw_dir = root / "data" / "raw"
    if synthetic or not any(raw_dir.glob("*.parquet")):
        if not synthetic:
            print("No data in data/raw yet -> using SYNTHETIC data. Run scripts/download_data.py first.")
        data = build_day_data(make_minute_bars(n_days=600, start="2021-01-04", seed=42))
        return data, data.daily_returns
    div_path = root / "data" / "SPY_dividends.csv"
    data = build_day_data(load_minute_bars(raw_dir), dividends=load_dividends(div_path) if div_path.exists() else None)
    bench_path = root / "data" / "SPY_daily_adj.parquet"
    bench = load_daily_benchmark(bench_path) if bench_path.exists() else data.daily_returns
    return data, bench.reindex(data.dates)


def make_periods(data: DayData, split: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Chronological train/test split. The train period starts after the burn-in."""
    split_ts = pd.Timestamp(split)
    return {"Train": (data.dates[BURN_IN_DAYS], split_ts - pd.Timedelta(days=1)),
            "Test": (split_ts, data.dates[-1])}


def evaluate(data: DayData, variants: dict[str, StrategyConfig], costs: dict[str, CostConfig],
             periods: dict[str, tuple], bench: pd.Series | None = None,
             include_benchmark: bool = True) -> tuple[pd.DataFrame, dict[tuple[str, str], BacktestResult]]:
    """Run every variant x cost scenario x period. Each period restarts with $100k.

    Returns a long table (one row per costs/strategy/period) and the full-sample results
    (keyed by (cost_name, variant_name)) for equity curves and trade-level analysis.
    """
    rows, full_results = [], {}
    first_day = min(s for s, _ in periods.values())
    for cost_name, cost in costs.items():
        for vname, cfg in variants.items():
            full_results[(cost_name, vname)] = run_backtest(data, cfg, cost, start=first_day)
            for pname, (s, e) in periods.items():
                res = run_backtest(data, cfg, cost, start=s, end=e)
                b = bench.loc[s:e] if bench is not None else None
                rows.append({"costs": cost_name, "strategy": vname, "period": pname,
                             **summarize(res.returns, b, res.trades)})
        if include_benchmark and bench is not None:
            for pname, (s, e) in periods.items():
                rows.append({"costs": cost_name, "strategy": "SPY buy & hold", "period": pname,
                             **summarize(bench.loc[s:e].fillna(0))})
    return pd.DataFrame(rows), full_results


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent / 2-decimal formatting for display."""
    out = df.copy()
    for c in out.columns:
        if c in PCT_COLUMNS:
            out[c] = out[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    return out


def summary_view(summary: pd.DataFrame, cost_name: str, columns: list[str] = MAIN_COLUMNS) -> pd.DataFrame:
    return summary[summary.costs == cost_name].set_index(["period", "strategy"])[columns]


def metric_bar_table(summary: pd.DataFrame, cost_name: str, order: list[str],
                     metrics=("sharpe", "ann_return", "ann_vol")) -> pd.DataFrame:
    """Reshape the summary for plotting.plot_metric_bars: rows = strategies, columns = (period, metric)."""
    tbl = (summary[summary.costs == cost_name]
           .pivot_table(index="strategy", columns="period", values=list(metrics))
           .swaplevel(axis=1).reindex(order))
    periods = list(dict.fromkeys(summary["period"]))
    return tbl[[(p, m) for p in periods for m in metrics]]


def cost_sensitivity(data: DayData, variants: dict[str, StrategyConfig], period: tuple,
                     slippages=(0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02),
                     commission: float = 0.0035, metric: str = "sharpe") -> pd.DataFrame:
    """Metric (default Sharpe) for a sweep of slippage values. Index = total cost per share."""
    s, e = period
    table = {name: [summarize(run_backtest(data, cfg, CostConfig(commission, sl), start=s, end=e).returns)[metric]
                    for sl in slippages] for name, cfg in variants.items()}
    out = pd.DataFrame(table, index=[commission + sl for sl in slippages])
    out.index.name = "cost_per_share"
    return out


def parameter_grid(data: DayData, base: StrategyConfig, costs: CostConfig, period: tuple,
                   lookbacks=(7, 14, 30, 60), multipliers=(0.8, 1.0, 1.2, 1.5),
                   metric: str = "sharpe") -> pd.DataFrame:
    """Metric for lookback x volatility multiplier. Use on the TRAIN period only."""
    s, e = period
    grid = {vm: [summarize(run_backtest(data, replace(base, lookback_days=lb, vol_multiplier=vm), costs,
                                        start=s, end=e).returns)[metric] for lb in lookbacks]
            for vm in multipliers}
    out = pd.DataFrame(grid, index=pd.Index(lookbacks, name="lookback_days"))
    out.columns.name = "vol_multiplier"
    return out
