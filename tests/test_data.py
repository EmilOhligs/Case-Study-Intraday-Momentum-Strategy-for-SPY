import numpy as np
import pandas as pd

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
    data = build_day_data(pd.concat([bars, pre]).sort_index())
    assert np.isclose(data.day_open[0], bars["open"].iloc[0])


def test_vwap_matches_manual_computation():
    bars = make_minute_bars(n_days=2, seed=3)
    data = build_day_data(bars)
    d0 = bars.iloc[:390]
    tp = (d0["high"] + d0["low"] + d0["close"]) / 3
    manual = (tp * d0["volume"]).cumsum() / d0["volume"].cumsum()
    assert np.allclose(data.vwap[0], manual.values)


def test_dividend_is_subtracted_from_previous_close_on_ex_date():
    import pandas as pd
    bars = make_minute_bars(n_days=5, seed=8)
    plain = build_day_data(bars)
    ex_date = plain.dates[3]
    adj = build_day_data(bars, dividends=pd.Series({ex_date: 1.5}))
    assert np.isclose(adj.prev_close[3], plain.prev_close[3] - 1.5)
    assert np.allclose(adj.prev_close[[1, 2, 4]], plain.prev_close[[1, 2, 4]])


def test_nyse_early_closes():
    import pandas as pd
    from intraday_momentum.data import nyse_early_closes
    days = pd.DatetimeIndex(["2025-11-28", "2024-12-24", "2023-07-03", "2025-07-03",
                             "2025-11-27", "2021-12-23", "2019-12-24", "2025-12-26"])
    assert set(nyse_early_closes(days).strftime("%Y-%m-%d")) == {"2025-11-28", "2024-12-24", "2023-07-03",
                                                                 "2025-07-03", "2019-12-24"}


def test_after_hours_bars_on_half_days_are_dropped():
    import pandas as pd
    bars = make_minute_bars(n_days=1, start="2025-11-28", seed=9)   # full 390 bars on a half day
    data = build_day_data(bars)
    assert data.n_bars[0] == 210
    assert np.isclose(data.day_close[0], bars.loc[:"2025-11-28 12:59", "close"].iloc[-1])
