import numpy as np
import pandas as pd

from conftest import bars_from_paths
from intraday_momentum import CostConfig, StrategyConfig, build_day_data, run_backtest
from intraday_momentum.features import FEATURES, build_features
from intraday_momentum.ml_strategy import cv_splits, positions_matrix, probs_to_positions
from intraday_momentum.synthetic import make_minute_bars


def test_features_have_no_look_ahead():
    """Changing prices AFTER a decision time must not change that row's features (only its label)."""
    bars = make_minute_bars(n_days=90, seed=11)
    base = build_features(build_day_data(bars))
    day = base.index.get_level_values("date")[-1]
    j = 149                                                        # decision at 12:00
    shocked = bars.copy()
    after = (shocked.index.normalize().tz_localize(None) == day) & \
            ((shocked.index.hour * 60 + shocked.index.minute - 570) > j)
    ramp = np.linspace(1.0, 1.03, after.sum())[:, None]            # a trend after the decision
    shocked.loc[after, ["open", "high", "low", "close"]] *= ramp
    new = build_features(build_day_data(shocked))
    assert np.allclose(base.loc[(day, j), FEATURES].values, new.loc[(day, j), FEATURES].values)
    assert not np.isclose(base.loc[(day, j), "fwd_ret"], new.loc[(day, j), "fwd_ret"])


def test_label_is_return_from_execution_to_close():
    data = build_day_data(make_minute_bars(n_days=90, seed=12))
    f = build_features(data)
    date, col = f.index[-1]
    i = int(np.flatnonzero(data.dates == date)[0])
    assert np.isclose(f.loc[(date, col), "fwd_ret"], data.day_close[i] / data.nxt_open[i, col] - 1)


def test_probs_to_positions_uses_symmetric_margin():
    p = pd.Series([0.60, 0.52, 0.50, 0.48, 0.40])
    assert list(probs_to_positions(p, 0.03)) == [1, 0, 0, 0, -1]


def test_cv_never_trains_on_validation_or_later_years():
    data = build_day_data(make_minute_bars(n_days=900, start="2016-01-04", seed=13))
    f = build_features(data)
    for year, train, val in cv_splits(f, "2016-01-01", [2018, 2019]):
        assert train.index.get_level_values("date").max() < pd.Timestamp(f"{year}-01-01")
        assert (val.index.get_level_values("date").year == year).all()


def test_target_positions_reproduce_hold_to_close(history):
    trend = 100 * (1 + np.linspace(0, 0.01, 391))
    data = build_day_data(bars_from_paths(history + [trend]))
    idx = pd.MultiIndex.from_tuples([(data.dates[-1], c) for c in range(29, 389, 30)], names=["date", "col"])
    target = positions_matrix(data, pd.Series(1, index=idx))
    costs = CostConfig(0.0035, 0.001)
    res = run_backtest(data, StrategyConfig(), costs, target_positions=target)
    shares = int(100_000 // data.day_open[-1])
    expected = shares * (trend[-1] - trend[30]) - 2 * shares * costs.per_share
    assert len(res.trades) == 1
    assert np.isclose(res.daily["gross_pnl"].iloc[-1] - res.daily["costs"].iloc[-1], expected)
