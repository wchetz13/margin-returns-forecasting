"""Walk-forward validation of return and volatility forecasts.

A single 80/20 split with a multi-step forecast (as in the notebooks) mostly
measures how fast an ARMA forecast decays to the unconditional mean. This
module instead evaluates *one-step-ahead* forecasts re-estimated through time
(expanding or rolling window), compares them with naive baselines, and tests
whether differences in accuracy are statistically meaningful.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

Forecaster = Callable[[pd.Series], float]


# --------------------------------------------------------------------------
# Synthetic data (used when market data is unavailable, and in the tests)
# --------------------------------------------------------------------------
def simulate_garch_returns(
    n: int = 2000,
    mu: float = 4e-4,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    ar1: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Simulate daily returns r_t = mu + ar1*(r_{t-1}-mu) + sigma_t z_t with GARCH(1,1) variance.

    Returns a business-day indexed Series.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    var = omega / (1.0 - alpha - beta)
    prev_eps = 0.0
    prev_r = mu
    for t in range(n):
        var = omega + alpha * prev_eps**2 + beta * var if t else var
        eps = np.sqrt(var) * z[t]
        r[t] = mu + ar1 * (prev_r - mu) + eps
        prev_eps, prev_r = eps, r[t]
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(r, index=idx, name="returns")


# --------------------------------------------------------------------------
# One-step-ahead forecasters (each maps a training window to a point forecast)
# --------------------------------------------------------------------------
def zero_forecast(train: pd.Series) -> float:
    """Random-walk-in-prices baseline: predict a zero return."""
    return 0.0


def mean_forecast(train: pd.Series) -> float:
    """Historical-mean baseline."""
    return float(train.mean())


def last_value_forecast(train: pd.Series) -> float:
    """Persistence baseline: tomorrow's return equals today's."""
    return float(train.iloc[-1])


def make_arima_forecaster(order: tuple[int, int, int] = (1, 0, 0)) -> Forecaster:
    """Return a forecaster that fits ARIMA(order) on the window and predicts one step."""
    from statsmodels.tsa.arima.model import ARIMA

    def forecast(train: pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(train.to_numpy(), order=order).fit()
            return float(fit.forecast(1)[0])

    return forecast


def walk_forward(
    series: pd.Series,
    forecaster: Forecaster,
    initial_window: int = 750,
    step: int = 1,
    rolling: bool = False,
) -> pd.Series:
    """One-step-ahead walk-forward forecasts.

    For each evaluation date t (every ``step``-th observation after the first
    ``initial_window``), ``forecaster`` is re-fit on data strictly before t
    (expanding, or the trailing ``initial_window`` if ``rolling``) and predicts
    the return at t. No information from t onwards is used, so there is no
    look-ahead. Larger ``step`` thins the evaluation grid to save refit time.
    """
    if initial_window >= len(series):
        raise ValueError("initial_window must be shorter than the series")
    out: dict[pd.Timestamp, float] = {}
    for t in range(initial_window, len(series), step):
        start = t - initial_window if rolling else 0
        out[series.index[t]] = forecaster(series.iloc[start:t])
    return pd.Series(out, name="forecast")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    hit_rate: float
    oos_r2: float


def evaluate(actual: pd.Series, forecast: pd.Series, benchmark: pd.Series | None = None) -> Metrics:
    """Accuracy metrics on the common index.

    ``oos_r2`` is the Campbell-Thompson out-of-sample R^2 against ``benchmark``
    (default: zero forecast): 1 - SSE_model / SSE_benchmark.
    """
    a = actual.loc[forecast.index].to_numpy()
    f = forecast.to_numpy()
    b = np.zeros_like(a) if benchmark is None else benchmark.loc[forecast.index].to_numpy()
    err = a - f
    nonzero = a != 0
    return Metrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        hit_rate=float(np.mean(np.sign(f[nonzero]) == np.sign(a[nonzero]))) if np.any(f != 0) else float("nan"),
        oos_r2=float(1.0 - np.sum(err**2) / np.sum((a - b) ** 2)),
    )


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test of equal squared-error accuracy (Harvey small-sample corrected).

    Returns (statistic, two-sided p-value). Negative statistic: model 1 is more accurate.
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0 + 2.0 * sum(
        (1.0 - k / h) * np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, h)
    )
    dm = dbar / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def compare_models(
    series: pd.Series,
    forecasters: dict[str, Forecaster],
    initial_window: int = 750,
    step: int = 5,
    rolling: bool = False,
    reference: str = "zero",
) -> pd.DataFrame:
    """Walk-forward every forecaster and tabulate metrics and DM tests vs ``reference``."""
    fcs = {k: walk_forward(series, f, initial_window, step, rolling) for k, f in forecasters.items()}
    ref = fcs[reference]
    ref_err = series.loc[ref.index].to_numpy() - ref.to_numpy()
    rows = {}
    for name, fc in fcs.items():
        m = evaluate(series, fc, benchmark=ref)
        err = series.loc[fc.index].to_numpy() - fc.to_numpy()
        dm, p = (np.nan, np.nan) if name == reference else diebold_mariano(err, ref_err)
        rows[name] = {"MAE": m.mae, "RMSE": m.rmse, "Hit rate": m.hit_rate,
                      "OOS R2 vs ref": m.oos_r2, "DM stat": dm, "DM p-value": p}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Volatility forecasts
# --------------------------------------------------------------------------
def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss, mean of RV/F - log(RV/F) - 1 (robust to noisy proxies, zero iff exact)."""
    ratio = np.asarray(realized_var) / np.asarray(forecast_var)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def garch_one_step_variance(train: pd.Series) -> float:
    """GARCH(1,1) one-step-ahead conditional variance (in return units squared)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(train * 100, vol="Garch", p=1, q=1).fit(disp="off")
        return float(res.forecast(horizon=1).variance.iloc[-1, 0]) / 1e4


def ewma_one_step_variance(train: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA variance forecast."""
    x = train.to_numpy()
    w = lam ** np.arange(len(x))[::-1]
    return float(np.sum(w * x**2) / np.sum(w))


def rolling_variance(train: pd.Series, window: int = 63) -> float:
    """Trailing-window sample second moment."""
    return float(np.mean(train.to_numpy()[-window:] ** 2))


def compare_volatility(
    series: pd.Series,
    forecasters: dict[str, Forecaster],
    initial_window: int = 750,
    step: int = 5,
) -> pd.DataFrame:
    """QLIKE and RMSE (vs squared-return proxy) of one-step variance forecasts."""
    proxy = series**2
    rows = {}
    for name, f in forecasters.items():
        fc = walk_forward(series, f, initial_window, step)
        rv = proxy.loc[fc.index].to_numpy()
        rows[name] = {"QLIKE": qlike(rv, fc.to_numpy()),
                      "Vol RMSE (bp)": 1e4 * float(np.sqrt(np.mean((np.sqrt(rv) - np.sqrt(fc.to_numpy())) ** 2)))}
    return pd.DataFrame(rows).T
