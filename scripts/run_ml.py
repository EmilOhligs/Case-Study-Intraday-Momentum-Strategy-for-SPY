"""Own strategy: ML (logistic regression) long/short/flat model. Figures and ml_summary.md go to results/.

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

from intraday_momentum.diagnostics import pnl_by_side  # noqa: E402
from intraday_momentum.evaluation import (COST_SCENARIOS, OVERVIEW_METRICS, PAPER_VARIANTS, SUBPERIODS,  # noqa: E402
                                          always_long_returns, evaluate, format_table, inverse_vol_mix,
                                          load_project_data, make_periods, metric_bar_table, subperiod_table,
                                          summary_view)
from intraday_momentum.metrics import sharpe_ratio  # noqa: E402
from intraday_momentum.ml_strategy import ML_VARIANTS, ml_pipeline  # noqa: E402
from intraday_momentum.plotting import plot_equity_curves, plot_metric_bars, plot_risk_return  # noqa: E402

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

    run = ml_pipeline(data, periods, COST_SCENARIOS)   # CV years: 2018-2021 on real data
    all_paper_summary, paper_full = evaluate(data, PAPER_VARIANTS, COST_SCENARIOS, periods, bench)
    paper_summary = all_paper_summary[all_paper_summary.strategy != "Base: opposite-band stop"]
    summary = pd.concat([paper_summary, run.summary], ignore_index=True)

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
        ml_result = run.full[("paper", "ML: logistic (1x)")]
        returns = {"Ext. 1: + band/VWAP stop": paper_full[("paper", "Ext. 1: + band/VWAP stop")].returns,
                   "ML: logistic (1x)": ml_result.returns,
                   "Control: always long 10:00 to close": always_long_returns(data, COST_SCENARIOS["paper"],
                                                                              periods["Train"][0])}
        sub = subperiod_table(returns, SUBPERIODS, bench)
        print("\nSub-periods (paper costs):\n", format_table(sub).to_string())
        f.write("\n### Sub-periods (paper costs)\n\n" + format_table(sub).to_markdown() + "\n")
        corr = pd.concat(returns, axis=1).loc[periods["Test"][0]:].corr().iloc[0, 1]
        print(f"\nCorrelation of daily returns ML vs Ext. 1 (test): {corr:.2f}")
        f.write(f"\nCorrelation of daily returns ML vs Ext. 1 (test period): {corr:.2f}\n")

        # controls: is the ML return just the intraday drift of SPY?
        test_start = periods["Test"][0]
        spy_intraday = pd.Series(data.day_close / data.day_open - 1, index=data.dates)
        corr_spy = ml_result.returns.loc[test_start:].corr(spy_intraday.loc[test_start:])
        sides = pnl_by_side(ml_result, test_start)
        weight, mix = inverse_vol_mix(returns["Ext. 1: + band/VWAP stop"], ml_result.returns, periods["Train"][1])
        print(f"Correlation of daily returns ML vs SPY open-to-close (test): {corr_spy:.2f}")
        print("\nML trades by side (test, gross):\n", sides.round(2).to_string())
        print(f"\nInverse-vol mix, weights from the train period: {weight:.0%} Ext. 1 / {1 - weight:.0%} ML, "
              f"test Sharpe {sharpe_ratio(mix):.2f}")
        f.write(f"\nCorrelation of daily returns ML vs SPY open-to-close (test period): {corr_spy:.2f}\n")
        f.write(f"\n### ML trades by side (test period, gross of costs)\n\n{sides.round(2).to_markdown()}\n")
        f.write(f"\n### Combination\n\nInverse-volatility mix with weights from the train period "
                f"({weight:.0%} Ext. 1, {1 - weight:.0%} ML): test Sharpe {sharpe_ratio(mix):.2f}\n")
        f.write(f"\n### AUC\n\n{run.auc.round(3).to_frame('AUC').to_markdown()}\n")

    # all implementations side by side
    all_summary = pd.concat([all_paper_summary, run.summary], ignore_index=True)
    all_order = [*PAPER_VARIANTS, *ML_VARIANTS, "SPY buy & hold"]
    overview = metric_bar_table(all_summary, "paper", all_order, OVERVIEW_METRICS)
    plot_metric_bars(overview, out / "metrics_overview.png", OVERVIEW_METRICS, ncols=2)
    plot_risk_return(overview, out / "risk_return.png")
    all_results = {**paper_full, **run.full}
    test_start = periods["Test"][0]
    curves = {name: all_results[("paper", name)].returns.loc[test_start:] for name in order[:-1]}
    curves["SPY buy & hold"] = bench.loc[test_start:]
    plot_equity_curves(curves, None, out / "ml_equity_test.png", "Test period: ML strategy vs paper rule vs SPY")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
