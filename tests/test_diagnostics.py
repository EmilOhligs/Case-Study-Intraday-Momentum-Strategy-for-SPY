import numpy as np

from conftest import bars_from_paths
from intraday_momentum import StrategyConfig, build_day_data, run_backtest
from intraday_momentum.diagnostics import false_breakout_stats, pnl_by_entry_time


def test_false_breakout_is_detected(history):
    # breakout above the band at 10:00, back to the open by 10:30 -> stopped out at the next decision
    up = 100 * (1 + np.concatenate([np.linspace(0, 0.004, 31), np.linspace(0.004, 0.0, 30), np.zeros(330)]))
    data = build_day_data(bars_from_paths(history + [up]))
    res = run_backtest(data, StrategyConfig(stop="band_vwap"))
    stats = false_breakout_stats(res)
    assert stats["n_trades"] == 1
    assert stats["false_breakout_rate"] == 1.0


def test_trade_held_to_close_is_not_a_false_breakout(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))
    data = build_day_data(bars_from_paths(history + [trend]))
    res = run_backtest(data, StrategyConfig(stop="band_vwap"))
    assert false_breakout_stats(res)["false_breakout_rate"] == 0.0
    assert list(pnl_by_entry_time(res).index) == ["10:00"]
