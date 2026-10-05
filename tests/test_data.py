import numpy as np

from intraday_momentum.data import build_day_data
from intraday_momentum.synthetic import make_minute_bars


def test_shapes_and_time_convention():
    bars = make_minute_bars(n_days=5, seed=1)
    data = build_day_data(bars)
    assert data.close.shape == (5, 390)
    day0 = bars[bars.index.normalize() == bars.index.normalize()[0]]
    # column j = bar starting at 09:30 + j  -> its close is the price at 09:30 + j + 1
    assert np.isclose(data.close[0, 29], day0["close"].iloc[29])
    # execution price for a decision at column j is the open of the next bar
    assert np.isclose(data.nxt_open[0, 29], day0["open"].iloc[30])
    assert np.isclose(data.day_open[0], day0["open"].iloc[0])
    assert np.isclose(data.prev_close[1], data.day_close[0])


def test_extended_hours_are_dropped():
    bars = make_minute_bars(n_days=3, seed=2)
    pre = bars.iloc[:5].copy()
    pre.index = pre.index - np.timedelta64(60, "m")          # 08:30 - 08:34, pre-market
    data = build_day_data(bars._append(pre).sort_index())
    assert np.isclose(data.day_open[0], bars["open"].iloc[0])


def test_vwap_matches_manual_computation():
    bars = make_minute_bars(n_days=2, seed=3)
    data = build_day_data(bars)
    d0 = bars.iloc[:390]
    tp = (d0["high"] + d0["low"] + d0["close"]) / 3
    manual = (tp * d0["volume"]).cumsum() / d0["volume"].cumsum()
    assert np.allclose(data.vwap[0], manual.values)
