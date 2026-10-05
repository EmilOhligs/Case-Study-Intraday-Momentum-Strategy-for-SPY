"""Run the full evaluation: paper replication (base + extensions), train/test split, costs, plots.

Usage:
    python scripts/run_backtest.py                 # real data from data/raw (see download_data.py)
    python scripts/run_backtest.py --synthetic     # smoke test on random-walk data
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_momentum import CostConfig, StrategyConfig, build_day_data, load_minute_bars, run_backtest, summarize
from intraday_momentum.data import load_daily_benchmark
from intraday_momentum.plotting import (plot_cost_sensitivity, plot_equity_curves, plot_metric_bars,
                                        plot_noise_area_day)
from intraday_momentum.synthetic import make_minute_bars

ROOT = Path(__file__).resolve().parents[1]

# --- Strategy variants (all other parameters = paper defaults) ---------------------------------
VARIANTS = {
    "Base: opposite-band stop": StrategyConfig(name="base", stop="opposite_band", sizing="full"),
    "Ext. 1: + band/VWAP stop": StrategyConfig(name="vwap", stop="band_vwap", sizing="full"),
    "Ext. 2: + vol targeting": StrategyConfig(name="paper", stop="band_vwap", sizing="vol_target"),
}

# --- Cost scenarios ------------------------------------------------------------------------------
COSTS = {
    "paper": CostConfig(commission_per_share=0.0035, slippage_per_share=0.001),
    "conservative": CostConfig(commission_per_share=0.0035, slippage_per_share=0.005),  # ~half the SPY spread
}

PCT = ["total_return", "ann_return", "ann_vol", "max_drawdown", "hit_ratio", "worst_day", "best_day", "alpha_ann"]


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in PCT:
            out[c] = out[c].map(lambda v: f"{v:.1%}")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:.2f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--benchmark", default=str(ROOT / "data" / "SPY_daily_adj.parquet"))
    p.add_argument("--split", default="2022-01-01", help="first day of the test period")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        bars = make_minute_bars(n_days=600, start="2021-01-04", seed=42)
        args.split = "2022-06-01"
        out = out / "synthetic"
        out.mkdir(parents=True, exist_ok=True)
    else:
        bars = load_minute_bars(args.data)
    data = build_day_data(bars)
    print(f"Loaded {len(data)} trading days: {data.dates[0].date()} -> {data.dates[-1].date()}")

    bench_path = Path(args.benchmark)
    bench = load_daily_benchmark(bench_path) if bench_path.exists() and not args.synthetic else data.daily_returns
    bench = bench.reindex(data.dates)

    first_day = data.dates[20]                         # burn-in for the 14-day lookbacks
    split = pd.Timestamp(args.split)
    periods = {"Train": (first_day, split - pd.Timedelta(days=1)), "Test": (split, data.dates[-1])}

    # ---- 1) Main results: every variant x period x cost scenario ----------------------------------
    rows, curves = [], {}
    for cost_name, costs in COSTS.items():
        for vname, cfg in VARIANTS.items():
            full = run_backtest(data, cfg, costs, start=first_day)
            if cost_name == "paper":
                curves[vname] = full.returns
            for pname, (s, e) in periods.items():
                res = run_backtest(data, cfg, costs, start=s, end=e)     # restart at $100k each period
                stats = summarize(res.returns, bench.loc[s:e], res.trades)
                rows.append({"costs": cost_name, "strategy": vname, "period": pname, **stats})
        b = bench.loc[first_day:]
        for pname, (s, e) in periods.items():
            rows.append({"costs": cost_name, "strategy": "SPY buy & hold", "period": pname,
                         **summarize(b.loc[s:e].fillna(0))})
    curves["SPY buy & hold"] = bench.loc[first_day:]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_ratio", "skew", "alpha_ann", "beta", "trades_per_day"]
    for cost_name in COSTS:
        view = summary[summary.costs == cost_name].set_index(["period", "strategy"])[cols]
        print(f"\n=== Costs: {cost_name} ({COSTS[cost_name].per_share:.4f} $/share) ===")
        print(fmt_table(view).to_string())
    with open(out / "summary.md", "w") as f:
        for cost_name in COSTS:
            view = summary[summary.costs == cost_name].set_index(["period", "strategy"])[cols]
            f.write(f"\n### Costs: {cost_name} ({COSTS[cost_name].per_share:.4f} $/share)\n\n")
            f.write(fmt_table(view).to_markdown() + "\n")

    # ---- 2) Plots ------------------------------------------------------------------------------------
    plot_equity_curves(curves, args.split, out / "equity_curves.png",
                       "Intraday momentum on SPY vs buy & hold (paper costs)")
    bars_tbl = (summary[summary.costs == "paper"]
                .pivot_table(index="strategy", columns="period", values=["sharpe", "ann_return", "ann_vol"])
                .swaplevel(axis=1).reindex(list(VARIANTS) + ["SPY buy & hold"]))
    bars_tbl = bars_tbl[[(pp, m) for pp in periods for m in ["sharpe", "ann_return", "ann_vol"]]]
    plot_metric_bars(bars_tbl, out / "metrics_train_vs_test.png")

    # ---- 3) Cost sensitivity on the test period -------------------------------------------------------
    grid = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
    s, e = periods["Test"]
    sens = pd.DataFrame({vname: [summarize(run_backtest(data, cfg, CostConfig(0.0035, sl), start=s, end=e).returns)["sharpe"]
                                 for sl in grid] for vname, cfg in VARIANTS.items()},
                        index=[0.0035 + sl for sl in grid])
    sens.index.name = "cost_per_share"
    sens.to_csv(out / "cost_sensitivity.csv")
    plot_cost_sensitivity(sens, out / "cost_sensitivity.png")

    # ---- 4) Robustness: parameter grid on the TRAIN period only ---------------------------------------
    s, e = periods["Train"]
    robust = pd.DataFrame(
        {vm: [summarize(run_backtest(data, replace(VARIANTS["Ext. 2: + vol targeting"], lookback_days=lb,
                                                   vol_multiplier=vm), COSTS["paper"], start=s, end=e).returns)["sharpe"]
              for lb in (7, 14, 30, 60)] for vm in (0.8, 1.0, 1.2, 1.5)},
        index=pd.Index((7, 14, 30, 60), name="lookback_days"))
    robust.columns.name = "vol_multiplier"
    robust.to_csv(out / "robustness_train_sharpe.csv")
    print("\nTrain-period Sharpe, lookback x VM (paper variant):\n", robust.round(2).to_string())

    # ---- 5) Example day ---------------------------------------------------------------------------------
    full = run_backtest(data, VARIANTS["Ext. 1: + band/VWAP stop"], COSTS["paper"], start=first_day)
    best = full.returns.idxmax()
    plot_noise_area_day(data, str(best.date()), out / "noise_area_example.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
