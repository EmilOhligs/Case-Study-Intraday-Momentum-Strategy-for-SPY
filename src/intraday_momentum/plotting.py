"""Figures for the presentation. Every function returns the figure and optionally saves it to `path`."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

from .data import DayData
from .signals import noise_area

def _finish(fig: plt.Figure, path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160)
    return fig


COLORS = ["#1f3b73", "#2a9d8f", "#e76f51", "#8d99ae", "#f4a261", "#6a4c93"]


def plot_equity_curves(curves: dict[str, pd.Series], split_date: str | None = None, path: Path | None = None,
                       title: str = "Equity curves") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (name, r), c in zip(curves.items(), COLORS):
        ax.plot((1 + r.fillna(0)).cumprod() * 100_000, label=name, color=c, lw=1.6)
    if split_date:
        ax.axvline(pd.Timestamp(split_date), color="k", ls="--", lw=1)
        ax.text(pd.Timestamp(split_date), ax.get_ylim()[1], "  test period →", va="top", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value ($, log scale)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


METRIC_LABELS = {"sharpe": "Sharpe ratio", "ann_return": "Annualized return", "ann_vol": "Annualized volatility",
                 "max_drawdown": "Max drawdown", "hit_ratio": "Hit ratio"}
PERIOD_COLORS = ["#8d99ae", "#1f3b73"]


def plot_metric_bars(summary: pd.DataFrame, path: Path | None = None,
                     metrics=("sharpe", "ann_return", "ann_vol"), ncols: int | None = None) -> plt.Figure:
    """summary: rows = strategies, columns = MultiIndex (period, metric). One panel per metric."""
    periods = list(dict.fromkeys(summary.columns.get_level_values(0)))
    ncols = ncols or len(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    width = max(4.2, 0.9 * len(summary)) * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 4.2 * nrows), squeeze=False)
    x = np.arange(len(summary))
    w = 0.8 / len(periods)
    for ax, m in zip(axes.flat, metrics):
        for k, (p, c) in enumerate(zip(periods, PERIOD_COLORS)):
            vals = summary[(p, m)].values
            bars = ax.bar(x + (k - (len(periods) - 1) / 2) * w, vals, w, label=p, color=c)
            fmt = "{:.2f}" if m == "sharpe" else "{:.0%}"
            ax.bar_label(bars, [fmt.format(v).replace("-0%", "0%") for v in vals], fontsize=7)
        if m != "sharpe":
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, summary.index, rotation=25, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.grid(axis="y", alpha=0.3)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    return _finish(fig, path)


def plot_risk_return(summary: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Annualized return vs. volatility, one panel per period. summary: as for plot_metric_bars.

    Strategies on the same dashed line have the same return per unit of risk, so leverage moves a
    strategy along a line and only a better signal moves it to a steeper one.
    """
    periods = list(dict.fromkeys(summary.columns.get_level_values(0)))
    fig, axes = plt.subplots(1, len(periods), figsize=(5.5 * len(periods), 4.6), sharex=True, sharey=True,
                             squeeze=False)
    vol_max = summary.xs("ann_vol", axis=1, level=1).max().max() * 1.15
    for ax, p in zip(axes.flat, periods):
        for ratio in (0.5, 1.0):
            ax.plot([0, vol_max], [0, ratio * vol_max], color="grey", ls="--", lw=0.8)
            ax.text(vol_max, ratio * vol_max, f" return / vol = {ratio}", fontsize=7, color="grey", va="center")
        for (name, row), c in zip(summary[p].iterrows(), COLORS):
            ax.scatter(row["ann_vol"], row["ann_return"], color=c, s=70, zorder=3, label=name)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(0, vol_max * 1.3)
        ax.set_title(f"{p} period")
        ax.set_xlabel("Annualized volatility")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Annualized return")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    return _finish(fig, path)


def plot_noise_area_day(data: DayData, date: str, path: Path | None = None, lookback: int = 14,
                        vm: float = 1.0, trades: pd.DataFrame | None = None) -> plt.Figure:
    i = int(np.flatnonzero(data.dates == pd.Timestamp(date))[0])
    ub, lb = noise_area(data, lookback, vm)
    t = pd.date_range(pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=31), periods=390, freq="min")
    n = data.n_bars[i]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(t[:n], lb[i, :n], ub[i, :n], color="#f4e3a1", alpha=0.7, label="Noise Area")
    ax.plot(t[:n], data.close[i, :n], color="k", lw=1, label="SPY")
    ax.plot(t[:n], data.vwap[i, :n], color="#e76f51", ls=":", lw=1.3, label="VWAP")
    for j in range(29, n - 1, 30):
        ax.axvline(t[j], color="grey", lw=0.4, alpha=0.5)
    if trades is not None and not trades.empty:
        day = trades[trades["date"] == pd.Timestamp(date)]
        for _, tr in day.iterrows():
            color = "#2a9d8f" if tr["side"] == 1 else "#e76f51"
            ax.scatter(t[int(tr["entry_col"])], tr["entry_px"], marker="^" if tr["side"] == 1 else "v",
                       color=color, s=60, zorder=5)
            ax.scatter(t[int(tr["exit_col"])], tr["exit_px"], marker="x", color=color, s=50, zorder=5)
    ax.set_title(f"Noise Area on {date}")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_cost_sensitivity(table: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """table: index = cost per share ($), columns = strategy names, values = Sharpe."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in zip(table.columns, COLORS):
        ax.plot(table.index, table[name], marker="o", label=name, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Total cost per share traded ($)")
    ax.set_ylabel("Sharpe ratio (test period)")
    ax.set_title("Cost sensitivity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return _finish(fig, path)
