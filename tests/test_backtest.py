import numpy as np

from conftest import bars_from_paths, quiet_day
from intraday_momentum import CostConfig, StrategyConfig, build_day_data, run_backtest


def test_no_trades_without_breakout(history):
    data = build_day_data(bars_from_paths(history + [quiet_day()]))
    res = run_backtest(data, StrategyConfig(), CostConfig())
    assert res.trades.empty
    assert np.allclose(res.returns, 0)


def test_trend_day_gives_one_long_trade_with_exact_pnl(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))           # +1% straight line
    data = build_day_data(bars_from_paths(history + [trend]))
    costs = CostConfig(commission_per_share=0.0035, slippage_per_share=0.001)
    res = run_backtest(data, StrategyConfig(), costs, initial_capital=100_000)

    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["side"] == 1 and t["entry_col"] == 29                 # first decision: 10:00
    shares = int(100_000 // data.day_open[-1])
    entry, exit_ = trend[30], trend[-1]          # path[j] = open of bar j -> open of the 10:00 bar
    expected = shares * (exit_ - entry) - 2 * shares * costs.per_share
    assert np.isclose(res.daily["gross_pnl"].iloc[-1] - res.daily["costs"].iloc[-1], expected)
    assert np.isclose(res.returns.iloc[-1], expected / 100_000)


def test_vwap_stop_exits_on_reversal_but_opposite_band_holds(history):
    up = np.linspace(0, 0.01, 200)
    down = np.linspace(0.01, 0.0005, 191)                   # falls back near the open, stays above LB
    day = 100 * (1 + np.concatenate([up, down]))
    data = build_day_data(bars_from_paths(history + [day]))

    base = run_backtest(data, StrategyConfig(stop="opposite_band"))
    vwap = run_backtest(data, StrategyConfig(stop="band_vwap"))
    assert len(base.trades) == 1 and base.trades.iloc[0]["exit_col"] == 389   # held to the close
    assert vwap.trades.iloc[0]["exit_col"] < 389                              # stopped out intraday
    assert vwap.returns.iloc[-1] > base.returns.iloc[-1]


def test_vol_targeting_caps_leverage(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))
    data = build_day_data(bars_from_paths(history + [trend]))
    res = run_backtest(data, StrategyConfig(sizing="vol_target", max_leverage=4.0))
    assert np.isclose(res.daily["leverage"].iloc[-1], 4.0)          # tiny vol in quiet history -> cap


def test_costs_reduce_returns(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))
    data = build_day_data(bars_from_paths(history + [trend]))
    cheap = run_backtest(data, costs=CostConfig(0.0, 0.0, 0.0)).returns.sum()
    dear = run_backtest(data, costs=CostConfig(0.0035, 0.01)).returns.sum()
    assert cheap > dear


def test_minimum_commission_applies_to_small_orders(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))
    data = build_day_data(bars_from_paths(history + [trend]))
    costs = CostConfig(commission_per_share=0.0035, slippage_per_share=0.0, min_commission_per_order=0.35)
    res = run_backtest(data, costs=costs, initial_capital=5_000)    # 50 shares -> 0.175 < 0.35 minimum
    assert np.isclose(res.daily["costs"].iloc[-1], 2 * 0.35)
