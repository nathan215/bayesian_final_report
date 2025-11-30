import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

def download_stocks(tickers, start_date, end_date):
    """Download adjusted close prices and compute log returns."""
    prices = yf.download(tickers, start=start_date, end=end_date, 
                        progress=False)['Close']
    
    if len(tickers) == 1:
        prices = prices.to_frame(name=tickers[0])
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def train_test_split_timeseries(data, train_size, test_size):
    """Split time series data (no shuffle)."""
    train = data.iloc[:train_size]
    test = data.iloc[train_size:train_size + test_size]
    return train, test
