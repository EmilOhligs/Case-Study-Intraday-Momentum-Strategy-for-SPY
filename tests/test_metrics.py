import numpy as np
import pandas as pd

from intraday_momentum.metrics import alpha_beta, annualized_return, max_drawdown, sharpe_ratio


def _series(x):
    return pd.Series(x, index=pd.bdate_range("2024-01-01", periods=len(x)))


def test_sharpe_is_invariant_to_leverage():
    r = _series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    assert np.isclose(sharpe_ratio(r), sharpe_ratio(2 * r))


def test_annualized_return_of_constant_daily_return():
    r = _series(np.full(252, 0.001))
    assert np.isclose(annualized_return(r), 1.001 ** 252 - 1)


def test_max_drawdown():
    assert np.isclose(max_drawdown(_series([0.1, -0.5, 0.2])), 0.5)


def test_alpha_beta_recovers_known_coefficients():
    b = _series(np.random.default_rng(1).normal(0, 0.01, 1000))
    res = alpha_beta(0.0001 + 0.5 * b, b)
    assert np.isclose(res["beta"], 0.5)
    assert np.isclose(res["alpha_ann"], 0.0001 * 252)
