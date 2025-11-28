import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def train_test_split_timeseries(df, train_ratio=0.8, date_column=None):
    if date_column is not None:
        df = df.set_index(date_column)
    
    split_idx = int(len(df) * train_ratio)
    split_date = df.index[split_idx]
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train-Test Split:")
    print(f"  Train: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"  Test:  {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"  Train size: {len(train_df)} observations")
    print(f"  Test size: {len(test_df)} observations")
    return train_df, test_df, split_date


def extract_stock_returns(returns_df, stock_name):
    if stock_name not in returns_df.columns:
        raise ValueError(f"Stock {stock_name} not found. Available: {returns_df.columns.tolist()}")
    
    return returns_df[stock_name].values

def compute_realized_volatility(returns, window=20):

    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    realized_vol = returns.rolling(window=window).std().values
    return realized_vol


def standardize_returns(returns):
    mean = np.mean(returns)
    std = np.std(returns)
    standardized = (returns - mean) / std
    return standardized, mean, std


def safe_log_transform(values, epsilon=1e-8):
    return np.log(np.abs(values) + epsilon)


def create_lagged_features(returns, lags=[1, 5, 20]):
    df = pd.DataFrame({'return_t': returns})
    
    for lag in lags:
        df[f'return_t_minus_{lag}'] = returns.shift(lag)
    return df.dropna()


def plot_train_test_split(train_series, test_series, stock_name, figsize=(14, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_series.index, train_series.values, 'b-', label='Train', alpha=0.7)
    ax.plot(test_series.index, test_series.values, 'r-', label='Test', alpha=0.7)
    ax.axvline(train_series.index[-1], color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    ax.set_title(f'{stock_name}: Train-Test Split')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def print_data_summary(returns_df, stock_list=None):
    if stock_list is None:
        stock_list = returns_df.columns.tolist()
    print("RETURNS DATA SUMMARY")
    for stock in stock_list:
        returns = returns_df[stock].dropna()
        print(f"\n{stock}:")
        print(f"  N observations: {len(returns)}")
        print(f"  Mean:           {returns.mean():.6f}")
        print(f"  Std Dev:        {returns.std():.6f}")
        print(f"  Min:            {returns.min():.6f}")
        print(f"  Max:            {returns.max():.6f}")
        print(f"  Skewness:       {returns.skew():.4f}")
        print(f"  Kurtosis:       {returns.kurtosis():.4f}")


def save_results(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, 'model_results.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results saved to {filepath}")


def load_results(output_dir):
    filepath = os.path.join(output_dir, 'model_results.pkl')
    
    with open(filepath, 'rb') as f:
        results_dict = pickle.load(f)
    
    print(f"Results loaded from {filepath}")
    return results_dict


def create_results_directory(base_dir='./results'):
    import os
    
    dirs = {
        'base': base_dir,
        'ar1_freq': os.path.join(base_dir, 'ar1_frequentist'),
        'ar1_bayes': os.path.join(base_dir, 'ar1_bayesian'),
        'garch_freq': os.path.join(base_dir, 'garch_frequentist'),
        'garch_bayes': os.path.join(base_dir, 'garch_bayesian'),
        'sv_bayes': os.path.join(base_dir, 'sv_bayesian'),
        'figures': os.path.join(base_dir, 'figures'),
    }
    
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)
    
    print(f"Created results directory structure in {base_dir}")
    return dirs


def generate_forecast_index(test_dates, forecast_steps=1):
    if forecast_steps == 1:
        return test_dates
    else:
        return test_dates


def compare_parameter_estimates(freq_params, bayes_summary, param_names):
    comparison_data = []
    
    for param in param_names:
        if param in freq_params:
            freq_val = freq_params[param]
            
            if param in bayes_summary.index:
                bayes_row = bayes_summary.loc[param]
                bayes_mean = bayes_row.get('mean', np.nan)
                bayes_std = bayes_row.get('std', np.nan)
                bayes_hdi_lower = bayes_row.get('hdi_2.5%', np.nan)
                bayes_hdi_upper = bayes_row.get('hdi_97.5%', np.nan)
                
                comparison_data.append({
                    'Parameter': param,
                    'Freq_Estimate': freq_val,
                    'Bayes_Mean': bayes_mean,
                    'Bayes_Std': bayes_std,
                    'Bayes_HDI_Lower': bayes_hdi_lower,
                    'Bayes_HDI_Upper': bayes_hdi_upper,
                    'Difference': abs(freq_val - bayes_mean)
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_mcmc_diagnostics(idata, var_names=['mu', 'phi', 'sigma_sq'], 
                          stock_name='', save_path=None):
    import arviz as az
    import matplotlib.pyplot as plt
    
    axes_array = az.plot_trace(idata, var_names=var_names, kind='rank_bars')
    
    if isinstance(axes_array, plt.Axes): 
        fig = axes_array.figure
    else: 
        fig = axes_array.flatten()[0].figure
        
    fig.suptitle(f'{stock_name}: MCMC Trace Plot & Rank Plot', fontsize=14, y=1.00)
    
    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def print_convergence_diagnostics(idata, var_names=['mu', 'phi', 'sigma_sq']):
    import arviz as az
    import pandas as pd
    
    summary = az.summary(idata, var_names=var_names)  
    
    conv_df = summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']].copy()
    
    converged = (summary['r_hat'] < 1.01).all() and (summary['ess_bulk'] > 400).all()
    
    return conv_df, converged

def compare_freq_vs_bayes_parameters(stock_list, ar1_freq_params, ar1_bayes_df_simple, 
                                      use_absolute_diff=True):
    import pandas as pd
    import numpy as np
    
    compare_data = []
    for stock in stock_list:
        freq_mu = ar1_freq_params[stock]['mu']
        bayes_mu = ar1_bayes_df_simple.loc[stock, 'mu_mean']
        
        freq_phi = ar1_freq_params[stock]['phi']
        bayes_phi = ar1_bayes_df_simple.loc[stock, 'phi_mean']
        
        freq_sigma = ar1_freq_params[stock]['sigma']
        bayes_sigma = ar1_bayes_df_simple.loc[stock, 'sigma_mean']
        
        if use_absolute_diff:
            diff_mu = abs(freq_mu - bayes_mu)
            diff_phi = abs(freq_phi - bayes_phi)
            diff_sigma = abs(freq_sigma - bayes_sigma)
        else:
            diff_mu = 100 * abs(freq_mu - bayes_mu) / (abs(freq_mu) + 1e-5)
            diff_phi = 100 * abs(freq_phi - bayes_phi) / (abs(freq_phi) + 1e-5)
            diff_sigma = 100 * abs(freq_sigma - bayes_sigma) / freq_sigma
        
        compare_data.append({
            'Stock': stock,
            'Freq_mu': freq_mu,
            'Bayes_mu': bayes_mu,
            'Diff_mu': diff_mu,
            'Freq_phi': freq_phi,
            'Bayes_phi': bayes_phi,
            'Diff_phi': diff_phi,
            'Freq_sigma': freq_sigma,
            'Bayes_sigma': bayes_sigma,
            'Diff_sigma': diff_sigma
        })
    
    comparison_df = pd.DataFrame(compare_data).set_index('Stock')
    
    # Flag issues: absolute difference > threshold
    if use_absolute_diff:
        issues = comparison_df[(comparison_df['Diff_mu'] > 0.002) |  # > 0.2%
                               (comparison_df['Diff_phi'] > 0.05) |   # > 5% movement
                               (comparison_df['Diff_sigma'] > 0.005)]  # > 0.5% volatility
    else:
        issues = comparison_df[(comparison_df['Diff_mu'] > 20) | 
                               (comparison_df['Diff_phi'] > 20) | 
                               (comparison_df['Diff_sigma'] > 20)]
    
    return comparison_df, issues
