"""Figures for the presentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import DayData  # noqa: E402
from .signals import noise_area  # noqa: E402

COLORS = ["#1f3b73", "#2a9d8f", "#e76f51", "#8d99ae", "#f4a261", "#6a4c93"]


def plot_equity_curves(curves: dict[str, pd.Series], split_date: str | None, path: Path, title: str) -> None:
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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, path: Path, metrics=("sharpe", "ann_return", "ann_vol")) -> None:
    """summary: rows = strategies, columns = MultiIndex (period, metric)."""
    labels = {"sharpe": "Sharpe ratio", "ann_return": "Annualized return", "ann_vol": "Annualized volatility"}
    periods = list(dict.fromkeys(summary.columns.get_level_values(0)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    x = np.arange(len(summary))
    w = 0.8 / len(periods)
    for ax, m in zip(axes, metrics):
        for k, (p, c) in enumerate(zip(periods, ["#8d99ae", "#1f3b73"])):
            vals = summary[(p, m)].values
            bars = ax.bar(x + (k - (len(periods) - 1) / 2) * w, vals, w, label=p, color=c)
            fmt = "{:.2f}" if m == "sharpe" else "{:.0%}"
            ax.bar_label(bars, [fmt.format(v) for v in vals], fontsize=7)
        ax.set_xticks(x, summary.index, rotation=25, ha="right", fontsize=8)
        ax.set_title(labels.get(m, m))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_noise_area_day(data: DayData, date: str, path: Path, lookback: int = 14, vm: float = 1.0) -> None:
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
    ax.set_title(f"Noise Area on {date}")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cost_sensitivity(table: pd.DataFrame, path: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
