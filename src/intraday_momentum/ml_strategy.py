"""Own strategy: a simple ML model that decides long / short / flat at the paper's decision times.

Model:      logistic regression (L2, standardised features) for P(return from now to the close > 0)
Position:   +1 if p > 0.5 + margin,  -1 if p < 0.5 - margin,  0 otherwise   (re-evaluated every 30 min)
Execution:  identical to the paper replication (next-bar open, flat at the close, same costs and sizing)
Selection:  C and margin chosen by expanding-window cross-validation by year, on the training period only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import BacktestResult, run_backtest
from .config import CostConfig, StrategyConfig
from .data import DayData
from .features import FEATURES, FEATURES_WITH_PAPER_SIGNAL, build_features
from .metrics import summarize


@dataclass(frozen=True)
class MLConfig:
    C: float = 1.0                         # inverse L2 strength (smaller = stronger regularisation)
    margin: float = 0.02                   # no-trade zone around p = 0.5
    features: tuple[str, ...] = tuple(FEATURES)


def make_model(C: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=1000))])


def fit(feats: pd.DataFrame, cfg: MLConfig) -> Pipeline:
    model = make_model(cfg.C)
    model.fit(feats[list(cfg.features)].values, feats["label"].values)
    return model


def predict_proba(model: Pipeline, feats: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    return pd.Series(model.predict_proba(feats[list(cfg.features)].values)[:, 1], index=feats.index, name="p_up")


def probs_to_positions(p: pd.Series, margin: float) -> pd.Series:
    pos = np.where(p > 0.5 + margin, 1, np.where(p < 0.5 - margin, -1, 0))
    return pd.Series(pos, index=p.index, name="position")


def positions_matrix(data: DayData, positions: pd.Series) -> np.ndarray:
    """(days x 390) matrix for run_backtest(target_positions=...); NaN where the model has no view."""
    m = np.full((len(data), 390), np.nan)
    row = pd.Series(np.arange(len(data)), index=data.dates)
    dates = positions.index.get_level_values("date")
    cols = positions.index.get_level_values("col").values
    m[row.loc[dates].values, cols] = positions.values
    return m


def coefficients(model: Pipeline, cfg: MLConfig) -> pd.Series:
    """Coefficients on standardised features: sign = direction of the effect, size = importance."""
    return pd.Series(model.named_steps["logit"].coef_[0], index=list(cfg.features)).sort_values(key=np.abs, ascending=False)


def backtest_ml(data: DayData, model: Pipeline, feats: pd.DataFrame, cfg: MLConfig, costs: CostConfig,
                start, end, sizing: str = "full") -> BacktestResult:
    """Predict on [start, end] and run the standard backtest engine with these positions."""
    f = feats[(feats.index.get_level_values("date") >= pd.Timestamp(start))
              & (feats.index.get_level_values("date") <= pd.Timestamp(end))]
    pos = probs_to_positions(predict_proba(model, f, cfg), cfg.margin)
    strat = StrategyConfig(name="ml", sizing=sizing)
    return run_backtest(data, strat, costs, start=start, end=end, target_positions=positions_matrix(data, pos))


def cv_splits(feats: pd.DataFrame, train_start, val_years: list[int], train_end=None):
    """Expanding window by calendar year: train on [train_start, Y-1], validate on year Y (never after train_end)."""
    dates = feats.index.get_level_values("date")
    last = pd.Timestamp(train_end) if train_end is not None else dates.max()
    for year in val_years:
        train = feats[(dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(f"{year}-01-01"))]
        val = feats[(dates >= pd.Timestamp(f"{year}-01-01")) & (dates <= min(pd.Timestamp(f"{year}-12-31"), last))]
        yield year, train, val


def default_val_years(feats: pd.DataFrame, train_start, train_end) -> list[int]:
    """All calendar years of the training period except the first two (which are only used for fitting)."""
    dates = feats.index.get_level_values("date")
    years = sorted(set(dates[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))].year))
    return years[2:] or years[1:]


def expanding_window_cv(data: DayData, feats: pd.DataFrame, train_start, val_years: list[int],
                        C_grid=(0.01, 0.1, 1.0), margin_grid=(0.0, 0.01, 0.02, 0.04),
                        costs: CostConfig = CostConfig(), features: tuple[str, ...] = tuple(FEATURES),
                        train_end=None) -> pd.DataFrame:
    """For each validation year Y: fit on [train_start, Y-1], evaluate on Y. Never touches the test period.

    Returns one row per (C, margin, year) with Sharpe after costs, AUC and trading activity.
    """
    rows = []
    for year, tr, va in cv_splits(feats, train_start, val_years, train_end):
        for C in C_grid:
            cfg = MLConfig(C=C, features=features)
            model = fit(tr, cfg)
            auc = roc_auc_score(va["label"], predict_proba(model, va, cfg))
            for m in margin_grid:
                cfg_m = MLConfig(C=C, margin=m, features=features)
                va_end = va.index.get_level_values("date").max()
                res = backtest_ml(data, model, va, cfg_m, costs, f"{year}-01-01", va_end)
                st = summarize(res.returns, trades=res.trades)
                rows.append({"C": C, "margin": m, "year": year, "sharpe": st["sharpe"], "ann_return": st["ann_return"],
                             "auc": auc, "trades_per_day": st.get("trades_per_day", 0.0),
                             "time_in_market": float((res.daily["n_trades"] > 0).mean())})
    return pd.DataFrame(rows)


def select_params(cv: pd.DataFrame, features: tuple[str, ...] = tuple(FEATURES)) -> tuple[MLConfig, pd.DataFrame]:
    """Pick (C, margin) with the highest mean validation Sharpe across years."""
    table = cv.groupby(["C", "margin"]).agg(mean_sharpe=("sharpe", "mean"), min_sharpe=("sharpe", "min"),
                                            mean_auc=("auc", "mean"), trades_per_day=("trades_per_day", "mean"))
    C, m = table["mean_sharpe"].idxmax()
    return MLConfig(C=float(C), margin=float(m), features=features), table.sort_values("mean_sharpe", ascending=False)


ML_VARIANTS = {"ML: logistic (1x)": "full", "ML: logistic + vol targeting": "vol_target"}


def evaluate_ml(data: DayData, model: Pipeline, feats: pd.DataFrame, cfg: MLConfig, periods: dict,
                costs: dict[str, CostConfig]) -> tuple[pd.DataFrame, dict[tuple[str, str], BacktestResult]]:
    """Same output format as evaluation.evaluate(), so ML and paper variants can be concatenated.

    Note: the model is fitted on the train period, so the 'Train' rows are in-sample.
    """
    rows, full = [], {}
    first, last = min(s for s, _ in periods.values()), max(e for _, e in periods.values())
    for cost_name, cost in costs.items():
        for name, sizing in ML_VARIANTS.items():
            full[(cost_name, name)] = backtest_ml(data, model, feats, cfg, cost, first, last, sizing)
            for pname, (s, e) in periods.items():
                res = backtest_ml(data, model, feats, cfg, cost, s, e, sizing)
                rows.append({"costs": cost_name, "strategy": name, "period": pname,
                             **summarize(res.returns, trades=res.trades)})
    return pd.DataFrame(rows), full


@dataclass
class MLRun:
    feats: pd.DataFrame
    cv: pd.DataFrame                    # all (C, margin, year) validation results, feature set A
    cv_table: pd.DataFrame              # aggregated over years, sorted by mean Sharpe
    feature_set_comparison: pd.DataFrame
    cfg: MLConfig
    model: Pipeline
    coefficients: pd.Series
    auc: pd.Series                      # train (in-sample) and test AUC
    summary: pd.DataFrame
    full: dict


def ml_pipeline(data: DayData, periods: dict, costs: dict[str, CostConfig],
                val_years: tuple[int, ...] | None = None) -> MLRun:
    """End-to-end: features -> CV on the train period -> final fit on train -> evaluation on all periods."""
    train_start, train_end = periods["Train"]
    feats = build_features(data)
    dates = feats.index.get_level_values("date")
    cv_costs = costs["paper"]
    val_years = list(val_years) if val_years is not None else default_val_years(feats, train_start, train_end)

    # 1) feature-set choice (A: own features, B: + the paper's discrete signal) - train period only
    comparison = {}
    for name, fs in {"A: 7 features": tuple(FEATURES), "B: A + paper signal": tuple(FEATURES_WITH_PAPER_SIGNAL)}.items():
        cv_fs = expanding_window_cv(data, feats, train_start, val_years, costs=cv_costs, features=fs,
                                     train_end=train_end)
        _, tbl = select_params(cv_fs, fs)
        comparison[name] = tbl.iloc[0].rename(f"{name} (best C={tbl.index[0][0]}, margin={tbl.index[0][1]})")
        if name.startswith("A"):
            cv_a = cv_fs
    comparison = pd.DataFrame(comparison.values())
    best_set = tuple(FEATURES) if comparison.iloc[0]["mean_sharpe"] >= comparison.iloc[1]["mean_sharpe"] \
        else tuple(FEATURES_WITH_PAPER_SIGNAL)
    cv = cv_a if best_set == tuple(FEATURES) else expanding_window_cv(data, feats, train_start, val_years,
                                                                      costs=cv_costs, features=best_set,
                                                                      train_end=train_end)
    cfg, cv_table = select_params(cv, best_set)

    # 2) final fit on the whole train period, then one evaluation
    train = feats[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))]
    test = feats[dates >= periods["Test"][0]]
    model = fit(train, cfg)
    auc = pd.Series({"train (in-sample)": roc_auc_score(train["label"], predict_proba(model, train, cfg)),
                     "test": roc_auc_score(test["label"], predict_proba(model, test, cfg))})
    summary, full = evaluate_ml(data, model, feats, cfg, periods, costs)
    return MLRun(feats, cv, cv_table, comparison, cfg, model, coefficients(model, cfg), auc, summary, full)
