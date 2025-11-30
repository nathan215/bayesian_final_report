# src/ar_models.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import time

def get_ar_prior(prior_set='medium'):
    """Define priors for AR(1) Bayesian"""
    priors = {
        'weak': {
            'phi_mean': 0.0,
            'phi_std': 1.0,
            'sigma2_alpha': 2.0,
            'sigma2_beta': 1.0,
            'description': 'Weak (diffuse)'
        },
        'medium': {
            'phi_mean': 0.3,
            'phi_std': 0.2,
            'sigma2_alpha': 3.0,
            'sigma2_beta': 0.5,
            'description': 'Medium (moderate belief in mean-reversion)'
        },
        'informative': {
            'phi_mean': 0.35,
            'phi_std': 0.1,
            'sigma2_alpha': 5.0,
            'sigma2_beta': 0.3,
            'description': 'Informative (strong belief in φ≈0.35)'
        }
    }
    return priors[prior_set]

def fit_ar_frequentist(train_data):
    """
    Fit AR(1) using OLS
    Returns dict with phi, sigma2, standard errors, etc.
    """
    X = train_data[:-1].reshape(-1, 1)
    y = train_data[1:]
    X = np.column_stack([np.ones(len(X)), X])
    
    # OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    constant = beta[0]
    phi = beta[1]
    residuals = y - X @ beta
    sigma2 = (residuals.T @ residuals) / (len(y) - 2)
    
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    
    # Information criteria
    ll = -0.5 * len(y) * np.log(2 * np.pi * sigma2) - (residuals.T @ residuals) / (2 * sigma2)
    aic = -2 * ll + 2 * 2
    bic = -2 * ll + 2 * np.log(len(y))
    
    return {
        'constant': float(constant),
        'phi': float(phi),
        'se_constant': float(se[0]),
        'se_phi': float(se[1]),
        'sigma2': float(sigma2),
        'residuals': residuals,
        'log_likelihood': float(ll),
        'aic': float(aic),
        'bic': float(bic),
        'method': 'freq'
    }

def gibbs_ar1(y, prior, n_samples=5000, n_burnin=1000):
    """
    Gibbs sampler for AR(1) model
    Returns: phi_samples, sigma2_samples
    """
    n = len(y)
    y_lag = y[:-1]
    y_curr = y[1:]
    
    phi_samples = np.zeros(n_samples)
    sigma2_samples = np.zeros(n_samples)
    
    phi = 0.1
    sigma2 = np.var(y)
    
    phi_mean = prior['phi_mean']
    phi_std = prior['phi_std']
    alpha = prior['sigma2_alpha']
    beta = prior['sigma2_beta']
    
    for i in range(n_samples + n_burnin):
        # Sample phi | sigma2, y
        post_var_inv = (y_lag.T @ y_lag) / sigma2 + 1 / (phi_std ** 2)
        post_var = 1 / post_var_inv
        post_mean = post_var * ((y_lag.T @ y_curr) / sigma2 + phi_mean / (phi_std ** 2))
        phi = np.random.normal(post_mean, np.sqrt(post_var))
        
        # Sample sigma2 | phi, y
        residuals = y_curr - phi * y_lag
        alpha_post = alpha + n / 2
        beta_post = beta + np.sum(residuals ** 2) / 2
        sigma2 = np.random.gamma(alpha_post, 1 / beta_post)
        
        if i >= n_burnin:
            phi_samples[i - n_burnin] = phi
            sigma2_samples[i - n_burnin] = sigma2
    
    return phi_samples, sigma2_samples

def fit_ar_bayesian(train_data, prior_set='medium', n_samples=5000, n_burnin=1000):
    """
    Fit AR(1) using Bayesian Gibbs sampler
    Returns dict with posterior samples and summary stats
    """
    prior = get_ar_prior(prior_set)
    
    start_time = time.time()
    phi_samples, sigma2_samples = gibbs_ar1(train_data, prior, n_samples, n_burnin)
    runtime = time.time() - start_time
    
    phi_mean = phi_samples.mean()
    phi_std = phi_samples.std()
    sigma2_mean = sigma2_samples.mean()
    sigma2_std = sigma2_samples.std()
    
    # MCMC diagnostics
    acf_phi = acf(phi_samples, nlags=1)[1]
    acf_sigma2 = acf(sigma2_samples, nlags=1)[1]
    
    # Effective sample size (simple approximation)
    ess_phi = n_samples / (1 + 2 * acf_phi) if acf_phi < 1 else n_samples
    ess_sigma2 = n_samples / (1 + 2 * acf_sigma2) if acf_sigma2 < 1 else n_samples
    
    return {
        'phi_mean': float(phi_mean),
        'phi_std': float(phi_std),
        'phi_samples': phi_samples,
        'phi_quantile_025': float(np.percentile(phi_samples, 2.5)),
        'phi_quantile_975': float(np.percentile(phi_samples, 97.5)),
        'sigma2_mean': float(sigma2_mean),
        'sigma2_std': float(sigma2_std),
        'sigma2_samples': sigma2_samples,
        'acf_phi': float(acf_phi),
        'acf_sigma2': float(acf_sigma2),
        'ess_phi': float(ess_phi),
        'ess_sigma2': float(ess_sigma2),
        'runtime': float(runtime),
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'prior_set': prior_set,
        'method': 'bayes'
    }


def forecast_ar_multihorizon(last_train, test_data, phi, horizons=[1, 5, 22]):
    """
    Generate multi-horizon forecasts
    
    horizons: list of forecast lengths
    Returns: dict with forecasts for each horizon
    """
    forecasts = {}
    
    for h in horizons:
        if h == 1:
            # 1-step: use last training observation
            fc = np.array([phi * last_train])
        else:
            # h-step: recursive forecasting
            fc = np.zeros(h)
            fc[0] = phi * last_train
            for t in range(1, h):
                fc[t] = phi * fc[t-1]
        
        # Truncate to actual test length
        actual = test_data[:h]
        mse = np.mean((actual - fc[:len(actual)]) ** 2)
        mae = np.mean(np.abs(actual - fc[:len(actual)]))
        
        forecasts[f'h_{h}'] = {
            'forecast': fc[:len(actual)],
            'actual': actual,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse))
        }
    
    return forecasts

def gibbs_ar1_hierarchical(data_dict, n_samples=5000, n_burnin=1000):
    """
    Gibbs sampler for hierarchical AR(1)
    
    data_dict: {'AAPL': array, 'MSFT': array, ...}
    
    Returns:
      - Individual phi samples for each stock
      - Group-level hyperparameter samples
    """
    stocks = list(data_dict.keys())
    K = len(stocks)  # number of stocks
    
    # Initialize
    phi_samples = {stock: np.zeros(n_samples) for stock in stocks}
    sigma2_samples = {stock: np.zeros(n_samples) for stock in stocks}
    mu_phi_samples = np.zeros(n_samples)
    tau_phi_samples = np.zeros(n_samples)
    
    # Initial values
    phi = {stock: 0.3 for stock in stocks}
    sigma2 = {stock: np.var(data_dict[stock]) for stock in stocks}
    mu_phi = 0.3
    tau_phi = 0.1
    
    # Gibbs loop
    for i in range(n_samples + n_burnin):
        
        # Sample individual phi | sigma2, mu_phi, tau_phi
        for stock in stocks:
            y = data_dict[stock]
            y_lag = y[:-1]
            y_curr = y[1:]
            n = len(y_curr)
            
            # Posterior precision and mean
            post_var_inv = (y_lag.T @ y_lag) / sigma2[stock] + 1 / (tau_phi ** 2)
            post_var = 1 / post_var_inv
            post_mean = post_var * ((y_lag.T @ y_curr) / sigma2[stock] + mu_phi / (tau_phi ** 2))
            
            phi[stock] = np.random.normal(post_mean, np.sqrt(post_var))
        
        # Sample individual sigma2 | phi, y
        for stock in stocks:
            y = data_dict[stock]
            y_lag = y[:-1]
            y_curr = y[1:]
            n = len(y_curr)
            
            residuals = y_curr - phi[stock] * y_lag
            alpha_post = 2 + n / 2
            beta_post = 1 + np.sum(residuals ** 2) / 2
            
            sigma2[stock] = np.random.gamma(alpha_post, 1 / beta_post)
        
        # Sample group mean mu_phi | phi, tau_phi
        phi_vals = np.array([phi[s] for s in stocks])
        post_var_mu = 1 / (K / (tau_phi ** 2) + 1 / 10 ** 2)  # prior variance = 10
        post_mean_mu = post_var_mu * (np.sum(phi_vals) / (tau_phi ** 2))
        mu_phi = np.random.normal(post_mean_mu, np.sqrt(post_var_mu))
        
        # Sample group SD tau_phi | phi, mu_phi
        phi_vals = np.array([phi[s] for s in stocks])
        sum_sq = np.sum((phi_vals - mu_phi) ** 2)
        alpha_tau = 2 + K / 2
        beta_tau = 1 + sum_sq / 2
        tau_phi = np.sqrt(np.random.gamma(alpha_tau, 1 / beta_tau))
        
        if i >= n_burnin:
            idx = i - n_burnin
            for stock in stocks:
                phi_samples[stock][idx] = phi[stock]
                sigma2_samples[stock][idx] = sigma2[stock]
            mu_phi_samples[idx] = mu_phi
            tau_phi_samples[idx] = tau_phi
    
    return {
        'phi_samples': phi_samples,
        'sigma2_samples': sigma2_samples,
        'mu_phi_samples': mu_phi_samples,
        'tau_phi_samples': tau_phi_samples,
        'n_samples': n_samples,
        'n_burnin': n_burnin
    }

def fit_ar_hierarchical(data_dict, n_samples=5000, n_burnin=1000):
    """
    Fit hierarchical AR(1) model to multiple stocks
    
    Returns dict with posterior summaries for each stock
    """
    start_time = time.time()
    samples = gibbs_ar1_hierarchical(data_dict, n_samples, n_burnin)
    runtime = time.time() - start_time
    
    stocks = list(data_dict.keys())
    
    results = {}
    for stock in stocks:
        phi_samp = samples['phi_samples'][stock]
        sigma2_samp = samples['sigma2_samples'][stock]
        
        results[stock] = {
            'phi_mean': float(phi_samp.mean()),
            'phi_std': float(phi_samp.std()),
            'phi_samples': phi_samp,
            'phi_quantile_025': float(np.percentile(phi_samp, 2.5)),
            'phi_quantile_975': float(np.percentile(phi_samp, 97.5)),
            'sigma2_mean': float(sigma2_samp.mean()),
            'sigma2_std': float(sigma2_samp.std()),
            'sigma2_samples': sigma2_samp,
            'acf_phi': float(acf(phi_samp, nlags=1)[1]),
            'acf_sigma2': float(acf(sigma2_samp, nlags=1)[1]),
        }
    
    # Group-level
    results['group'] = {
        'mu_phi_samples': samples['mu_phi_samples'],
        'mu_phi_mean': float(samples['mu_phi_samples'].mean()),
        'mu_phi_std': float(samples['mu_phi_samples'].std()),
        'tau_phi_samples': samples['tau_phi_samples'],
        'tau_phi_mean': float(samples['tau_phi_samples'].mean()),
        'tau_phi_std': float(samples['tau_phi_samples'].std()),
    }
    
    results['meta'] = {
        'runtime': float(runtime),
        'n_samples': n_samples,
        'n_burnin': n_burnin,
        'method': 'bayes_hierarchical'
    }
    
    return results
