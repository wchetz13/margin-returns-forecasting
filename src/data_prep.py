import yfinance as yf
import pandas as pd
import numpy as np

def get_returns(ticker="QQQ"):
    data = yf.download(ticker, start="2018-01-01")

    # Handle both old and new yfinance formats
    if "Adj Close" in data.columns:
        price = data["Adj Close"]
    else:
        price = data["Close"]

    data["returns"] = price.pct_change()
    return data.dropna()


def create_margin_series():
    dates = pd.date_range(start="2018-01-01", periods=60, freq="ME")
    
    trend = 0.30 + 0.001 * np.arange(len(dates))
    noise = np.random.normal(0, 0.01, len(dates))
    
    df = pd.DataFrame({
        "ds": dates,
        "y": trend + noise
    })
    
    return df