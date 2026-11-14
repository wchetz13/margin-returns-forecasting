"""Run the walk-forward study and write tables/figures to ``outputs/``.

Usage:  python -m src.run_validation [TICKER]

With a ticker it downloads daily returns via yfinance; if the download fails
(or no ticker is given) it falls back to simulated AR(1)-GARCH(1,1) returns so
the study is always reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import validation as v

OUT = Path(__file__).resolve().parent.parent / "outputs"


def load_returns(ticker: str | None) -> tuple[pd.Series, str]:
    if ticker:
        try:
            from src.data_prep import get_returns

            r = get_returns(ticker)["returns"].dropna()
            r = r.squeeze()
            if len(r) > 1000:
                return r, f"{ticker} (yfinance)"
        except Exception as exc:  # network / schema problems
            print(f"Download failed ({exc!r}); using simulated data.")
    return v.simulate_garch_returns(2500, ar1=0.05, seed=42), "simulated AR(1)-GARCH(1,1)"


def main(ticker: str | None = None) -> None:
    r, label = load_returns(ticker)
    print(f"Data: {label}, n={len(r)}\n")

    mean_models = {
        "zero": v.zero_forecast,
        "hist. mean": v.mean_forecast,
        "persistence": v.last_value_forecast,
        "AR(1)": v.make_arima_forecaster((1, 0, 0)),
        "ARMA(2,2)": v.make_arima_forecaster((2, 0, 2)),
    }
    ret_tab = v.compare_models(r, mean_models, initial_window=750, step=5)
    print("Return forecasts (one-step walk-forward, expanding window)")
    print(ret_tab.round(5).to_string(), "\n")

    vol_models = {
        "rolling 63d": v.rolling_variance,
        "EWMA 0.94": v.ewma_one_step_variance,
        "GARCH(1,1)": v.garch_one_step_variance,
    }
    vol_tab = v.compare_volatility(r, vol_models, initial_window=750, step=5)
    print("Volatility forecasts")
    print(vol_tab.round(4).to_string())

    OUT.mkdir(exist_ok=True)
    ret_tab.round(5).to_csv(OUT / "walk_forward_returns.csv")
    vol_tab.round(4).to_csv(OUT / "walk_forward_volatility.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ret_tab["RMSE"].mul(1e4).plot.bar(ax=axes[0], color="#4c72b0", rot=30)
    axes[0].set_title("Return forecast RMSE (bp, lower is better)")
    axes[0].set_ylim(ret_tab["RMSE"].min() * 1e4 * 0.98, ret_tab["RMSE"].max() * 1e4 * 1.01)
    vol_tab["QLIKE"].plot.bar(ax=axes[1], color="#dd8452", rot=30)
    axes[1].set_title("Variance forecast QLIKE (lower is better)")
    fig.suptitle(f"Walk-forward comparison - {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "walk_forward_comparison.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
