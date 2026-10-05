import numpy as np

from intraday_momentum.data import build_day_data
from intraday_momentum.signals import average_move_from_open, noise_area, trailing_daily_vol
from intraday_momentum.synthetic import make_minute_bars


def test_sigma_has_no_look_ahead():
    """Changing prices on day k must not change sigma on days <= k (only from k+1 on)."""
    bars = make_minute_bars(n_days=40, seed=4)
    base = average_move_from_open(build_day_data(bars), 14)

    k = 30
    day_k = bars.index.normalize() == bars.index.normalize().unique()[k]
    shocked = bars.copy()
    shocked.loc[day_k, ["open", "high", "low", "close"]] *= np.linspace(1, 1.05, day_k.sum())[:, None]
    after = average_move_from_open(build_day_data(shocked), 14)

    assert np.allclose(base[: k + 1], after[: k + 1], equal_nan=True)
    assert not np.allclose(base[k + 1], after[k + 1])


def test_sigma_is_mean_absolute_move_of_previous_days():
    bars = make_minute_bars(n_days=20, seed=5)
    data = build_day_data(bars)
    sigma = average_move_from_open(data, 14)
    t, j = 16, 100
    manual = np.mean(np.abs(data.close[t - 14:t, j] / data.day_open[t - 14:t] - 1))
    assert np.isclose(sigma[t, j], manual)


def test_gap_adjustment_widens_band_on_the_gap_side():
    bars = make_minute_bars(n_days=20, seed=6)
    data = build_day_data(bars)
    ub, lb = noise_area(data, 14, 1.0)
    t = 18
    anchor_up = max(data.day_open[t], data.prev_close[t])
    anchor_dn = min(data.day_open[t], data.prev_close[t])
    sigma = average_move_from_open(data, 14)[t]
    assert np.allclose(ub[t], anchor_up * (1 + sigma))
    assert np.allclose(lb[t], anchor_dn * (1 - sigma))


def test_daily_vol_uses_only_past_returns():
    data = build_day_data(make_minute_bars(n_days=30, seed=7))
    vol = trailing_daily_vol(data, 14)
    r = data.daily_returns.values
    assert np.isclose(vol[20], np.std(r[6:20], ddof=1))
