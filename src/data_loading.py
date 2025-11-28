import pandas as pd
from pathlib import Path

def load_log_returns(filepath='data/processed/log_returns.csv'):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def load_prices(filepath='data/processed/aligned_prices.csv'):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def get_summary_stats(returns_df):
    return pd.DataFrame({
        'Mean': returns_df.mean(),
        'Std Dev': returns_df.std(),
        'Skewness': returns_df.skew(),
        'Kurtosis': returns_df.kurtosis(),
        'Min': returns_df.min(),
        'Max': returns_df.max(),
        'Obs': returns_df.shape[0]
    })

def get_data_info(returns_df):
    print("DATA SUMMARY")
    print(f"Shape: {returns_df.shape[0]} obs × {returns_df.shape[1]} stocks")
    print(f"Tickers: {', '.join(returns_df.columns)}")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
