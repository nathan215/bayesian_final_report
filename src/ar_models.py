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
            'sigma2_alpha': 1.0,
            'sigma2_beta': 1.0,
            'description': 'Weak (diffuse)'
        },
        'medium': {
            'phi_mean': 0.0,
            'phi_std': 0.2,
            'sigma2_alpha': 10.0,
            'sigma2_beta': 10.0,
            'description': 'Medium'
        },
        'informative': {
            'phi_mean': 0.0,
            'phi_std': 0.1,
            'sigma2_alpha': 100.0,
            'sigma2_beta': 100.0,
            'description': 'Informative '
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

def gibbs_ar1_hierarchical(data_dict, n_samples=5000, n_burnin=1000):
    """
    Gibbs sampler for hierarchical AR(1) with INFORMATIVE priors
    
    Equivalent to: φ_k ~ N(0, 0.1²) for each stock
                   σ²_k ~ Gamma(100, 100)
    
    data_dict: {'AAPL': array, 'MSFT': array, ...}
    """
    stocks = list(data_dict.keys())
    K = len(stocks)
    
    # Initialize
    phi_samples = {stock: np.zeros(n_samples) for stock in stocks}
    sigma2_samples = {stock: np.zeros(n_samples) for stock in stocks}
    mu_phi_samples = np.zeros(n_samples)
    tau_phi_samples = np.zeros(n_samples)
    
    # Initial values
    phi = {stock: 0.0 for stock in stocks}
    sigma2 = {stock: np.var(data_dict[stock]) for stock in stocks}
    mu_phi = 0.0
    tau_phi = 0.05  # ← TIGHTER: was 0.1, now 0.05 (strong pooling)
    
    # PRIOR HYPERPARAMETERS (informative)
    prior_mu_phi_mean = 0.0
    prior_mu_phi_var = 0.01  # ← TIGHT: was 100, now 0.01 (equiv to σ=0.1)
    prior_tau_phi_shape = 100.0  # ← STRONG: was 2, now 100 (tight around mode)
    prior_tau_phi_rate = 100.0   # ← STRONG: was 1, now 100
    
    # ====================================================================
    # ALTERNATIVE: Directly control σ²_k prior to match non-hierarchical
    # ====================================================================
    prior_sigma2_alpha = 100.0  # ← MATCH non-hierarchical
    prior_sigma2_beta = 100.0   # ← MATCH non-hierarchical
    
    print(f"Informative Hierarchical AR(1) Priors:")
    print(f"  φ group mean μ_φ: N(0, {prior_mu_phi_var})")
    print(f"  φ group SD τ_φ²: Gamma({prior_tau_phi_shape}, {prior_tau_phi_rate})")
    print(f"  σ²_k: Gamma({prior_sigma2_alpha}, {prior_sigma2_beta})")
    print(f"  Initial τ_φ: {tau_phi}\n")
    
    # Gibbs loop
    for i in range(n_samples + n_burnin):
        
        # ================================================================
        # STEP 1: Sample individual φ_k | σ²_k, μ_φ, τ_φ
        # ================================================================
        for stock in stocks:
            y = data_dict[stock]
            y_lag = y[:-1]
            y_curr = y[1:]
            n = len(y_curr)
            
            # Posterior precision and mean (from normal likelihood + normal prior)
            post_var_inv = (y_lag.T @ y_lag) / sigma2[stock] + 1 / (tau_phi ** 2)
            post_var = 1 / post_var_inv
            post_mean = post_var * ((y_lag.T @ y_curr) / sigma2[stock] + mu_phi / (tau_phi ** 2))
            
            phi[stock] = np.random.normal(post_mean, np.sqrt(post_var))
        
        # ================================================================
        # STEP 2: Sample individual σ²_k | φ_k, y (with informative prior)
        # ================================================================
        for stock in stocks:
            y = data_dict[stock]
            y_lag = y[:-1]
            y_curr = y[1:]
            n = len(y_curr)
            
            residuals = y_curr - phi[stock] * y_lag
            
            # INFORMATIVE PRIOR: Gamma(alpha_prior, beta_prior)
            # Posterior: Gamma(alpha_prior + n/2, beta_prior + sum_sq/2)
            alpha_post = prior_sigma2_alpha + n / 2
            beta_post = prior_sigma2_beta + np.sum(residuals ** 2) / 2
            
            sigma2[stock] = np.random.gamma(alpha_post, 1 / beta_post)
        
        # ================================================================
        # STEP 3: Sample group mean μ_φ | φ_k, τ_φ (with tight prior)
        # ================================================================
        phi_vals = np.array([phi[s] for s in stocks])
        
        # Posterior: inverse variance weighted average
        post_var_mu = 1 / (K / (tau_phi ** 2) + 1 / prior_mu_phi_var)
        post_mean_mu = post_var_mu * (np.sum(phi_vals) / (tau_phi ** 2) + 
                                       prior_mu_phi_mean / prior_mu_phi_var)
        mu_phi = np.random.normal(post_mean_mu, np.sqrt(post_var_mu))
        
        # ================================================================
        # STEP 4: Sample group SD τ_φ | φ_k, μ_φ (with tight prior)
        # ================================================================
        phi_vals = np.array([phi[s] for s in stocks])
        sum_sq = np.sum((phi_vals - mu_phi) ** 2)
        
        # INFORMATIVE HYPERPRIOR: Gamma(shape, rate) on τ_φ²
        # Posterior: Gamma(shape + K/2, rate + sum_sq/2)
        alpha_tau = prior_tau_phi_shape + K / 2
        beta_tau = prior_tau_phi_rate + sum_sq / 2
        
        tau_phi = np.sqrt(np.random.gamma(alpha_tau, 1 / beta_tau))
        
        # Store samples after burnin
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
        'n_burnin': n_burnin,
        'prior_config': {
            'prior_mu_phi_var': prior_mu_phi_var,
            'prior_tau_phi_shape': prior_tau_phi_shape,
            'prior_tau_phi_rate': prior_tau_phi_rate,
            'prior_sigma2_alpha': prior_sigma2_alpha,
            'prior_sigma2_beta': prior_sigma2_beta
        }
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

def forecast_ar_multihorizon_point(last_train, test_data, phi, horizons=[1, 5, 22]):
    forecasts = {}
    
    for h in horizons:
        if h == 1:
            fc = np.array([phi * last_train])
        else:
            fc = np.zeros(h)
            fc[0] = phi * last_train
            for t in range(1, h):
                fc[t] = phi * fc[t-1]
        
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


def forecast_ar_posterior_predictive(last_train, test_data, phi_samples, 
                                        sigma2_samples, horizons=[1, 5, 22], 
                                        n_posterior_max=1000, max_phi_abs=0.999):
    """
    FIXED VERSION: Numerical stability + Prediction Interval Score (PIS)
    
    SAFEGUARDS:
    1. Clip |φ| > 0.999 → 0.999 (prevent explosion)
    2. Cap sigma2 at 99th percentile 
    3. Limit recursion depth with damping for long horizons
    4. Use proper PIS scoring
    
    PIS Formula (95% intervals, α=0.05):
    PIS = (upper - lower) + (2/0.95) * [(lower - actual) if actual < lower else 0] 
                        + (2/0.95) * [(actual - upper) if actual > upper else 0]
    """
    n_posterior = min(len(phi_samples), n_posterior_max)  # Subsample if too many
    forecasts = {}
    
    # Preprocess: clip extreme parameters
    phi_clipped = np.clip(np.abs(phi_samples[:n_posterior]), 0, max_phi_abs)
    phi_clipped = np.sign(phi_samples[:n_posterior]) * phi_clipped
    sigma2_clipped = np.clip(sigma2_samples[:n_posterior], 
                           0, np.percentile(sigma2_samples, 99))
    
    for h in horizons:
        # Generate stable forecasts
        posterior_forecasts = np.zeros((n_posterior, h))
        
        for i in range(n_posterior):
            phi = phi_clipped[i]
            sigma2 = sigma2_clipped[i]
            
            current = last_train
            damping_factor = 1.0  # Decay for stability in long horizons
            
            for t in range(h):
                # Add damping for long horizons to prevent explosion
                if t > 10:  # After 10 steps, dampen phi
                    phi_effective = phi * 0.95 ** (t - 10)
                else:
                    phi_effective = phi
                
                noise = np.random.normal(0, np.sqrt(sigma2) * damping_factor)
                current = phi_effective * current + noise
                posterior_forecasts[i, t] = current
                
                # Additional safeguard: clip extreme values
                if abs(current) > 10:  # Arbitrary reasonable bound for returns
                    current *= 0.9  # Dampen
                    posterior_forecasts[i, t] = current
        
        # Compute prediction intervals from posterior
        forecast_mean = np.mean(posterior_forecasts, axis=0)
        ci_lower = np.percentile(posterior_forecasts, 2.5, axis=0)
        ci_upper = np.percentile(posterior_forecasts, 97.5, axis=0)
        
        # Truncate to test data length
        max_test_len = min(h, len(test_data))
        forecast_mean = forecast_mean[:max_test_len]
        ci_lower = ci_lower[:max_test_len]
        ci_upper = ci_upper[:max_test_len]
        actual = test_data[:max_test_len]
        
        # ====== NEW: PREDICTION INTERVAL SCORE (PIS) ======
        alpha = 0.05  # 95% intervals
        pis_scores = []
        
        for t in range(len(actual)):
            lower, upper, y = ci_lower[t], ci_upper[t], actual[t]
            
            if y < lower:
                pis = (upper - lower) + (2/0.95) * (lower - y)
            elif y > upper:
                pis = (upper - lower) + (2/0.95) * (y - upper)
            else:
                pis = (upper - lower)
            
            pis_scores.append(pis)
        
        mean_pis = float(np.mean(pis_scores))
        
        # Traditional metrics
        coverage = float(np.mean((actual >= ci_lower) & (actual <= ci_upper)))
        mse = float(np.mean((actual - forecast_mean) ** 2))
        
        forecasts[f'h_{h}'] = {
            'forecast_mean': forecast_mean.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'actual': actual.tolist(),
            'coverage': coverage,
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'interval_width': float(np.mean(ci_upper - ci_lower)),
            'pis': mean_pis,  # NEW: Prediction Interval Score
            'pis_per_step': pis_scores  # For diagnostics
        }
    
    return forecasts


def compute_pis_summary(forecasts_dict):
    """Extract PIS from forecasts for table"""
    pis_data = []
    for h_key in ['h_1', 'h_5', 'h_22']:
        if h_key in forecasts_dict:
            pis_data.append(forecasts_dict[h_key]['pis'])
    return pis_data
