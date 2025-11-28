import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

def download_and_process_stocks(tickers, start_date, end_date, output_dir='data/processed'):
    """Download and process stock data."""
    print("DATA PREPROCESSING: Stock Price Download and Cleaning")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading data for {len(tickers)} stocks...")
    raw_prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    print(f"Downloaded: {raw_prices.shape[0]} trading days × {raw_prices.shape[1]} stocks")
    
    initial_rows = len(raw_prices)
    aligned_prices = raw_prices.dropna()
    final_rows = len(aligned_prices)
    print(f"Rows after alignment: {final_rows} (dropped {initial_rows - final_rows} rows)")
    
    print(f"\nComputing log returns...")
    log_prices = np.log(aligned_prices)
    log_returns = log_prices.diff().dropna()
    log_returns.columns = [f'{col}_return' for col in log_returns.columns]
    
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Date range: {log_returns.index[0].date()} to {log_returns.index[-1].date()}")
    
    print(f"\nSaving processed data to {output_dir}/...")
    
    log_returns_path = Path(output_dir) / 'log_returns.csv'
    log_returns.to_csv(log_returns_path)
    print(f"Saved: {log_returns_path}")
    
    aligned_prices_path = Path(output_dir) / 'aligned_prices.csv'
    aligned_prices.to_csv(aligned_prices_path)
    print(f"Saved: {aligned_prices_path}")
    
    print("PREPROCESSING COMPLETE")

    return log_returns

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    log_returns = download_and_process_stocks(
        tickers=tickers,
        start_date='2020-11-24',
        end_date='2025-11-24',
        output_dir='data/processed'
    )