"""Intraday momentum strategy on SPY (Zarattini, Aziz & Barbon, 2024) - replication and extensions."""
from .config import CostConfig, StrategyConfig
from .data import DayData, build_day_data, load_minute_bars
from .backtest import BacktestResult, run_backtest
from .metrics import summarize

__all__ = ["CostConfig", "StrategyConfig", "DayData", "build_day_data", "load_minute_bars",
           "BacktestResult", "run_backtest", "summarize"]
