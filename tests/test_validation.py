import numpy as np
import pandas as pd
import pytest

from src import validation as v


@pytest.fixture(scope="module")
def r() -> pd.Series:
    return v.simulate_garch_returns(n=1500, seed=1)


def test_simulation_reproducible_and_shape():
    a = v.simulate_garch_returns(300, seed=3)
    b = v.simulate_garch_returns(300, seed=3)
    pd.testing.assert_series_equal(a, b)
    assert len(a) == 300 and a.index.is_monotonic_increasing


def test_garch_simulation_has_vol_clustering(r):
    x = r.to_numpy() ** 2
    assert np.corrcoef(x[1:], x[:-1])[0, 1] > 0.1


def test_walk_forward_has_no_lookahead(r):
    seen = []

    def spy(train: pd.Series) -> float:
        seen.append(train.index[-1])
        return 0.0

    fc = v.walk_forward(r, spy, initial_window=500, step=7)
    assert len(fc) == len(seen)
    for t, last in zip(fc.index, seen):
        assert last < t


def test_walk_forward_rolling_window_length(r):
    lens = []
    v.walk_forward(r, lambda s: lens.append(len(s)) or 0.0, 400, 50, rolling=True)
    assert set(lens) == {400}


def test_walk_forward_rejects_short_series(r):
    with pytest.raises(ValueError):
        v.walk_forward(r, v.zero_forecast, initial_window=len(r))


def test_evaluate_perfect_forecast(r):
    m = v.evaluate(r, r)
    assert m.mae == 0 and m.rmse == 0 and m.hit_rate == 1.0 and m.oos_r2 == 1.0


def test_evaluate_zero_forecast_has_zero_r2(r):
    m = v.evaluate(r, pd.Series(0.0, index=r.index))
    assert m.oos_r2 == pytest.approx(0.0)
    assert m.rmse == pytest.approx(np.sqrt(np.mean(r**2)))


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1, 2000)
    e_bad = rng.normal(0, 1.5, 2000)
    stat, p = v.diebold_mariano(e_good, e_bad)
    assert stat < 0 and p < 0.01


def test_diebold_mariano_equal_models_not_significant():
    rng = np.random.default_rng(1)
    stat, p = v.diebold_mariano(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert p > 0.01


def test_arima_beats_zero_when_returns_are_autocorrelated():
    s = v.simulate_garch_returns(2000, alpha=0.0, beta=0.0, ar1=0.3, seed=5)
    tab = v.compare_models(
        s, {"zero": v.zero_forecast, "AR(1)": v.make_arima_forecaster((1, 0, 0))},
        initial_window=600, step=10)
    assert tab.loc["AR(1)", "OOS R2 vs ref"] > 0.03
    assert tab.loc["AR(1)", "DM p-value"] < 0.05


def test_no_predictability_in_iid_returns():
    s = v.simulate_garch_returns(1500, alpha=0.0, beta=0.0, mu=0.0, seed=2)
    tab = v.compare_models(
        s, {"zero": v.zero_forecast, "AR(1)": v.make_arima_forecaster((1, 0, 0))},
        initial_window=600, step=10)
    assert tab.loc["AR(1)", "OOS R2 vs ref"] < 0.01


def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(0)
    rv = rng.chisquare(1, 5000) * 1e-4
    assert v.qlike(rv, rv) == pytest.approx(0.0)
    assert v.qlike(rv, np.full_like(rv, rv.mean())) > 0


def test_ewma_and_rolling_variance_constant_series():
    x = pd.Series(np.full(200, 0.01))
    assert v.ewma_one_step_variance(x) == pytest.approx(1e-4)
    assert v.rolling_variance(x) == pytest.approx(1e-4)


def test_garch_beats_constant_variance_on_garch_data():
    s = v.simulate_garch_returns(1600, alpha=0.1, beta=0.88, seed=4)
    tab = v.compare_volatility(
        s,
        {"const": lambda t: float(np.mean(t.to_numpy() ** 2)), "GARCH": v.garch_one_step_variance},
        initial_window=700, step=15)
    assert tab.loc["GARCH", "QLIKE"] < tab.loc["const", "QLIKE"]
