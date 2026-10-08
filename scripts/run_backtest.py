"""Run the full evaluation of the paper replication. Figures and summary.md go to results/, raw CSVs to results/tables/.

Usage:
    python scripts/run_backtest.py                 # real data (see scripts/download_data.py)
    python scripts/run_backtest.py --synthetic     # smoke test on random-walk data

The same steps can be run interactively in notebooks/backtest.ipynb.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: only save figures

from intraday_momentum.evaluation import (COST_SCENARIOS, PAPER_VARIANTS, cost_sensitivity, evaluate,  # noqa: E402
                                          format_table, load_project_data, make_periods, metric_bar_table,
                                          parameter_grid, summary_view)
from intraday_momentum.plotting import (plot_cost_sensitivity, plot_equity_curves, plot_metric_bars,  # noqa: E402
                                        plot_noise_area_day)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="2022-01-01", help="first day of the test period")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    data, bench = load_project_data(ROOT, synthetic=args.synthetic)
    split = "2022-06-01" if args.synthetic else args.split
    out = ROOT / "results" / ("synthetic" if args.synthetic else "")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(data)} trading days: {data.dates[0].date()} -> {data.dates[-1].date()}")

    periods = make_periods(data, split)
    summary, full = evaluate(data, PAPER_VARIANTS, COST_SCENARIOS, periods, bench)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "summary.csv", index=False)
    with open(out / "summary.md", "w") as f:
        for cost_name, cost in COST_SCENARIOS.items():
            table = format_table(summary_view(summary, cost_name))
            print(f"\n=== Costs: {cost_name} ({cost.per_share:.4f} $/share) ===\n{table.to_string()}")
            f.write(f"\n### Costs: {cost_name} ({cost.per_share:.4f} $/share)\n\n{table.to_markdown()}\n")

    curves = {name: full[("paper", name)].returns for name in PAPER_VARIANTS}
    curves["SPY buy & hold"] = bench.loc[periods["Train"][0]:]
    plot_equity_curves(curves, split, out / "equity_curves.png", "Intraday momentum on SPY vs buy & hold (paper costs)")
    order = list(PAPER_VARIANTS) + ["SPY buy & hold"]
    plot_metric_bars(metric_bar_table(summary, "paper", order), out / "metrics_train_vs_test.png")

    sens = cost_sensitivity(data, PAPER_VARIANTS, periods["Test"])
    sens.to_csv(tables / "cost_sensitivity.csv")
    plot_cost_sensitivity(sens, out / "cost_sensitivity.png")

    grid = parameter_grid(data, PAPER_VARIANTS["Ext. 2: + vol targeting"], COST_SCENARIOS["paper"], periods["Train"])
    grid.to_csv(tables / "robustness_train_sharpe.csv")
    print("\nTrain-period Sharpe, lookback x VM (paper variant):\n", grid.round(2).to_string())

    ext1 = full[("paper", "Ext. 1: + band/VWAP stop")]
    best_day = str(ext1.returns.idxmax().date())
    plot_noise_area_day(data, best_day, out / "noise_area_example.png", trades=ext1.trades)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
