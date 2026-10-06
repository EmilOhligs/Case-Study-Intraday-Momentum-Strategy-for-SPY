"""Own strategy: ML (logistic regression) long/short/flat model. Writes results/ml_*.

Usage:
    python scripts/run_ml.py              # real data
    python scripts/run_ml.py --synthetic  # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from intraday_momentum.evaluation import (COST_SCENARIOS, PAPER_VARIANTS, SUBPERIODS, evaluate, format_table,  # noqa: E402
                                          load_project_data, make_periods, metric_bar_table, subperiod_table,
                                          summary_view)
from intraday_momentum.ml_strategy import ML_VARIANTS, ml_pipeline  # noqa: E402
from intraday_momentum.plotting import plot_equity_curves, plot_metric_bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="2022-01-01")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    data, bench = load_project_data(ROOT, synthetic=args.synthetic)
    split = "2022-06-01" if args.synthetic else args.split
    out = ROOT / "results" / ("synthetic" if args.synthetic else "")
    out.mkdir(parents=True, exist_ok=True)
    periods = make_periods(data, split)

    run = ml_pipeline(data, periods, COST_SCENARIOS, bench)   # CV years: 2018-2021 on real data
    paper_summary, paper_full = evaluate(data, {k: PAPER_VARIANTS[k] for k in ["Ext. 1: + band/VWAP stop",
                                                                                 "Ext. 2: + vol targeting"]},
                                         COST_SCENARIOS, periods, bench)
    summary = pd.concat([paper_summary, run.summary], ignore_index=True)
    summary.to_csv(out / "ml_summary.csv", index=False)
    run.cv_table.to_csv(out / "ml_cv_table.csv")
    run.feature_set_comparison.to_csv(out / "ml_feature_set_comparison.csv")
    run.coefficients.to_csv(out / "ml_coefficients.csv")

    print("Feature-set comparison (CV on train):\n", run.feature_set_comparison.round(3).to_string())
    print("\nSelected:", run.cfg)
    print("\nCV table (top 5):\n", run.cv_table.head().round(3).to_string())
    print("\nCoefficients (standardised):\n", run.coefficients.round(3).to_string())
    print("\nAUC:\n", run.auc.round(3).to_string())
    order = ["Ext. 1: + band/VWAP stop", "Ext. 2: + vol targeting", *ML_VARIANTS, "SPY buy & hold"]
    with open(out / "ml_summary.md", "w") as f:
        for cost_name in COST_SCENARIOS:
            table = format_table(summary_view(summary, cost_name))
            print(f"\n=== Costs: {cost_name} ===\n{table.to_string()}")
            f.write(f"\n### Costs: {cost_name}\n\n{table.to_markdown()}\n")
        returns = {"Ext. 1: + band/VWAP stop": paper_full[("paper", "Ext. 1: + band/VWAP stop")].returns,
                   "ML: logistic (1x)": run.full[("paper", "ML: logistic (1x)")].returns}
        sub = subperiod_table(returns, SUBPERIODS, bench)
        print("\nSub-periods (paper costs):\n", format_table(sub).to_string())
        f.write("\n### Sub-periods (paper costs)\n\n" + format_table(sub).to_markdown() + "\n")
        corr = pd.concat(returns, axis=1).loc[periods["Test"][0]:].corr().iloc[0, 1]
        print(f"\nCorrelation of daily returns ML vs Ext. 1 (test): {corr:.2f}")
        f.write(f"\nCorrelation of daily returns ML vs Ext. 1 (test period): {corr:.2f}\n")

    plot_metric_bars(metric_bar_table(summary, "paper", order), out / "ml_metrics_train_vs_test.png")
    all_results = {**paper_full, **run.full}
    test_start = periods["Test"][0]
    curves = {name: all_results[("paper", name)].returns.loc[test_start:] for name in order[:-1]}
    curves["SPY buy & hold"] = bench.loc[test_start:]
    plot_equity_curves(curves, None, out / "ml_equity_test.png", "Test period: ML strategy vs paper rule vs SPY")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
