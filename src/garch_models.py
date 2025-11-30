import numpy as np
import pandas as pd
from scipy import stats
import time
from typing import Dict, Tuple, Any


def fit_garch_frequentist(returns: np.ndarray, max_iterations=1000, tol=1e-6) -> Dict[str, Any]:
    start_time = time.time()
    
    y = returns
    n = len(y)
    
    # Initialize parameters
    omega = np.var(y) * 0.05
    alpha = 0.1
    beta = 0.85
    
    # Initial variance
    h = np.ones(n) * np.var(y)
    
    loglik_old = -np.inf
    
    for iteration in range(max_iterations):
        # E-step: Compute conditional variances
        h[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
        
        # Avoid negative/zero variances
        h = np.maximum(h, 1e-8)
        
        # M-step: Update parameters via gradient ascent
        loglik = -0.5 * np.sum(np.log(h) + y**2 / h)
        
        if np.abs(loglik - loglik_old) < tol:
            break
        
        loglik_old = loglik
        
        # Gradient w.r.t. omega, alpha, beta
        residuals_sq = y**2
        
        # Simple gradient step
        grad_omega = -0.5 * np.sum(1/h - residuals_sq / h**2)
        grad_alpha = -0.5 * np.sum((residuals_sq[:-1] / h[1:]) * 
                                   (1/h[1:] - residuals_sq[1:] / h[1:]**2))
        grad_beta = -0.5 * np.sum((h[:-1] / h[1:]) * 
                                  (1/h[1:] - residuals_sq[1:] / h[1:]**2))
        
        # Update with damping
        learning_rate = 0.01
        omega = np.maximum(omega + learning_rate * grad_omega, 1e-6)
        alpha = np.clip(alpha + learning_rate * grad_alpha, 1e-4, 0.5)
        beta = np.clip(beta + learning_rate * grad_beta, 0.1, 0.99)
        
        # Ensure stationarity: α + β < 1
        if alpha + beta >= 1.0:
            scale = 0.99 / (alpha + beta)
            alpha *= scale
            beta *= scale
    
    runtime = time.time() - start_time
    
    return {
        'omega': float(omega),
        'alpha': float(alpha),
        'beta': float(beta),
        'sigma2': h[-1],
        'loglik': float(loglik),
        'convergence': iteration < max_iterations,
        'runtime': runtime,
    }



def get_garch_prior(prior_set: str = 'medium') -> Dict[str, Tuple]:
    priors = {
        'weak': {
            'omega': (1.0, 1.0),      # Exponential: mean=1
            'alpha': (1.0, 1.0),      # Beta(1,1) - Uniform on [0,1]
            'beta': (1.0, 1.0),
        },
        'medium': {
            'omega': (2.0, 0.1),      # Gamma: mean=0.2, variance=0.02
            'alpha': (2.0, 20.0),     # Beta: concentrated around 0.09
            'beta': (20.0, 3.0),      # Beta: concentrated around 0.87
        },
        'informative': {
            'omega': (3.0, 0.05),     # Tighter around 0.15
            'alpha': (2.5, 25.0),     # Beta: concentrated around 0.09
            'beta': (30.0, 5.0),      # Beta: concentrated around 0.86
        }
    }
    
    return priors[prior_set]


def fit_garch_bayesian(returns: np.ndarray, prior_set: str = 'medium',
                       n_samples: int = 5000, n_burnin: int = 1000) -> Dict[str, Any]:

    start_time = time.time()
    
    y = returns
    n = len(y)
    
    priors = get_garch_prior(prior_set)
    
    # Initialize
    omega = 0.05
    alpha = 0.1
    beta = 0.85
    h = np.ones(n) * np.var(y)
    
    # Storage
    samples_omega = np.zeros(n_samples)
    samples_alpha = np.zeros(n_samples)
    samples_beta = np.zeros(n_samples)
    
    n_total = n_burnin + n_samples
    
    for iteration in range(n_total):

        garch_param_denom = np.maximum(1 - alpha - beta, 1e-10) 
        

        scale_h0 = 2.0 / np.maximum((y[0]**2 + omega / garch_param_denom), 1e-10)
        
        sampled_gamma_h0 = np.random.gamma(0.5, scale_h0)
        h[0] = 1.0 / np.maximum(sampled_gamma_h0, 1e-15)

        for t in range(1, n):
            mu_h = omega + alpha * y[t-1]**2 + beta * h[t-1]
            
            safe_denominator = np.maximum((y[t]**2 + mu_h), 1e-10)
            scale_ht = 2.0 / safe_denominator
            
            sampled_gamma_ht = np.random.gamma(0.5, scale_ht)
            h[t] = 1.0 / np.maximum(sampled_gamma_ht, 1e-15)
    
            h = np.maximum(h, 1e-8) 
        
        # ----- Sample omega (via Exponential) -----
        prior_omega_shape, prior_omega_scale = priors['omega']
        # Likelihood contribution simplified
        omega = np.random.gamma(prior_omega_shape, prior_omega_scale)
        omega = np.maximum(omega, 1e-6)
        
        # ----- Sample alpha (via Beta) -----
        prior_alpha_a, prior_alpha_b = priors['alpha']
        alpha_prop = np.random.beta(prior_alpha_a, prior_alpha_b)
        if alpha_prop + beta < 1.0:  # Stationarity constraint
            loglik_curr = -0.5 * np.sum(np.log(h) + y**2 / h) 
            alpha = alpha_prop
        
        # ----- Sample beta (via Beta) -----
        prior_beta_a, prior_beta_b = priors['beta']
        beta_prop = np.random.beta(prior_beta_a, prior_beta_b)
        if alpha + beta_prop < 1.0:
            beta = beta_prop
        
        if iteration >= n_burnin:
            idx = iteration - n_burnin
            samples_omega[idx] = omega
            samples_alpha[idx] = alpha
            samples_beta[idx] = beta
    
    runtime = time.time() - start_time
    
    return {
        'omega_mean': float(np.mean(samples_omega)),
        'omega_std': float(np.std(samples_omega)),
        'alpha_mean': float(np.mean(samples_alpha)),
        'alpha_std': float(np.std(samples_alpha)),
        'beta_mean': float(np.mean(samples_beta)),
        'beta_std': float(np.std(samples_beta)),
        'samples_omega': samples_omega,
        'samples_alpha': samples_alpha,
        'samples_beta': samples_beta,
        'runtime': runtime,
    }


def fit_garch_hierarchical(data_dict: Dict[str, np.ndarray],
                          n_samples: int = 5000, n_burnin: int = 1000) -> Dict[str, Any]:
    """
    Fit hierarchical GARCH(1,1) with proper hyperpriors.
    """
    start_time = time.time()
    
    tickers = list(data_dict.keys())
    K = len(tickers)
    
    # Storage
    samples_mu_omega = np.zeros(n_samples)
    samples_mu_alpha = np.zeros(n_samples)
    samples_mu_beta = np.zeros(n_samples)
    samples_sigma_omega = np.zeros(n_samples) 
    samples_sigma_alpha = np.zeros(n_samples)
    samples_sigma_beta = np.zeros(n_samples)
    
    samples_omega = {t: np.zeros(n_samples) for t in tickers}
    samples_alpha = {t: np.zeros(n_samples) for t in tickers}
    samples_beta = {t: np.zeros(n_samples) for t in tickers}
    
    mu_alpha = 0.08
    mu_beta = 0.85
    mu_omega = 0.005 
    
    sigma_alpha = 0.03 
    sigma_beta = 0.05
    sigma_omega = 0.002 
    
    params = {t: {
        'omega': 0.005, 
        'alpha': 0.08,
        'beta': 0.85,
        'h': np.ones(len(data_dict[t])) * np.var(data_dict[t])
    } for t in tickers}
    
    n_total = n_burnin + n_samples
    
    for iteration in range(n_total):
        # ----- Update individual stocks -----
        for ticker in tickers:
            y = data_dict[ticker]
            n = len(y)
            h = params[ticker]['h']

            garch_param_denom_h0 = np.maximum(1 - mu_alpha - mu_beta, 1e-10) 
            

            scale_h0 = 2.0 / np.maximum(y[0]**2 + params[ticker]['omega'] / garch_param_denom_h0, 1e-10)
            h[0] = 1.0 / np.maximum(np.random.gamma(0.5, scale_h0), 1e-15)
            
            for t in range(1, n):
                mu_h = params[ticker]['omega'] + params[ticker]['alpha'] * y[t-1]**2 + \
                       params[ticker]['beta'] * h[t-1]
                scale_ht = 2.0 / np.maximum(y[t]**2 + mu_h, 1e-10)
                h[t] = 1.0 / np.maximum(np.random.gamma(0.5, scale_ht), 1e-15)
            
            params[ticker]['h'] = np.maximum(h, 1e-8)
            

            shape_omega = 2.0 + 0.5 * n 

            rate_omega = (1.0/mu_omega) + 0.5 * np.sum(1.0/h) 
            params[ticker]['omega'] = np.random.gamma(shape_omega, 1.0/rate_omega) 
            params[ticker]['omega'] = np.maximum(params[ticker]['omega'], 1e-9) 

            tau_alpha_eff = 1.0 / (sigma_alpha**2 + 1e-9) 
            alpha_prop = np.random.normal(
                (tau_alpha_eff * mu_alpha + params[ticker]['alpha'] * 1e-6) / (tau_alpha_eff + 1e-6), 
                1.0 / np.sqrt(tau_alpha_eff + 1e-6)
            )
            alpha_prop = np.clip(alpha_prop, 0.01, 0.5)
            if alpha_prop + params[ticker]['beta'] < 1.0:
                params[ticker]['alpha'] = alpha_prop
            
            tau_beta_eff = 1.0 / (sigma_beta**2 + 1e-9)
            beta_prop = np.random.normal(
                (tau_beta_eff * mu_beta + params[ticker]['beta'] * 1e-6) / (tau_beta_eff + 1e-6), 
                1.0 / np.sqrt(tau_beta_eff + 1e-6)
            )
            beta_prop = np.clip(beta_prop, 0.3, 0.95) 
            if params[ticker]['alpha'] + beta_prop < 1.0:
                params[ticker]['beta'] = beta_prop
        
        # ----- Update hyperparameters with proper priors -----
        omegas = np.array([params[t]['omega'] for t in tickers])
        alphas = np.array([params[t]['alpha'] for t in tickers])
        betas = np.array([params[t]['beta'] for t in tickers])
        
        mu_omega = np.random.normal(np.mean(omegas), 0.001) 
        mu_omega = np.clip(mu_omega, 1e-7, 0.05)

        mu_alpha = np.random.normal(np.mean(alphas), 0.01) 
        mu_alpha = np.clip(mu_alpha, 0.01, 0.3) 
        
        mu_beta = np.random.normal(np.mean(betas), 0.01)
        mu_beta = np.clip(mu_beta, 0.75, 0.95)  
        

        sigma_omega = np.sqrt(np.var(omegas) + 1e-6) 
        sigma_omega = np.clip(sigma_omega, 1e-7, 0.01) 

        sigma_alpha = np.sqrt(np.var(alphas) + 1e-5) 
        sigma_beta = np.sqrt(np.var(betas) + 1e-5) 
        
        # Store post-burnin
        if iteration >= n_burnin:
            idx = iteration - n_burnin
            samples_mu_omega[idx] = mu_omega # ADDED
            samples_sigma_omega[idx] = sigma_omega # ADDED
            samples_mu_alpha[idx] = mu_alpha
            samples_mu_beta[idx] = mu_beta
            samples_sigma_alpha[idx] = sigma_alpha
            samples_sigma_beta[idx] = sigma_beta
            
            for ticker in tickers:
                samples_omega[ticker][idx] = params[ticker]['omega']
                samples_alpha[ticker][idx] = params[ticker]['alpha']
                samples_beta[ticker][idx] = params[ticker]['beta']
    
    runtime = time.time() - start_time
    
    # Compile results
    result = {
        'meta': {
            'n_stocks': K,
            'n_samples': n_samples,
            'n_burnin': n_burnin,
            'runtime': runtime,
        },
        'group': {
            'mu_omega_mean': float(np.mean(samples_mu_omega)), # ADDED
            'mu_omega_std': float(np.std(samples_mu_omega)),   # ADDED
            'sigma_omega_mean': float(np.mean(samples_sigma_omega)), # ADDED
            'sigma_omega_std': float(np.std(samples_sigma_omega)), # ADDED
            'mu_alpha_mean': float(np.mean(samples_mu_alpha)),
            'mu_alpha_std': float(np.std(samples_mu_alpha)),
            'mu_beta_mean': float(np.mean(samples_mu_beta)),
            'mu_beta_std': float(np.std(samples_mu_beta)),
            'sigma_alpha_mean': float(np.mean(samples_sigma_alpha)),
            'sigma_beta_mean': float(np.mean(samples_sigma_beta)),
        }
    }
    
    for ticker in tickers:
        result[ticker] = {
            'omega_mean': float(np.mean(samples_omega[ticker])), # Individual omega mean
            'omega_std': float(np.std(samples_omega[ticker])),   # Individual omega std
            'alpha_mean': float(np.mean(samples_alpha[ticker])),
            'alpha_std': float(np.std(samples_alpha[ticker])),
            'beta_mean': float(np.mean(samples_beta[ticker])),
            'beta_std': float(np.std(samples_beta[ticker])),
            'samples_omega': samples_omega[ticker], # Individual omega chains
            'samples_alpha': samples_alpha[ticker],
            'samples_beta': samples_beta[ticker],
        }
    
    return result


def forecast_garch_rolling(train_returns: np.ndarray, test_returns: np.ndarray,
                          alpha: float, beta: float, omega: float,
                          horizons: list = [1, 5, 22]) -> Dict[str, float]:
    h = omega / (1 - alpha - beta)
    for ret in train_returns[-min(50, len(train_returns)):]:
        h = omega + alpha * ret**2 + beta * h
    
    mse_by_horizon = {}
    
    for h_idx in range(len(horizons)):
        horizon = horizons[h_idx]
        squared_errors = []
        
        for i in range(len(test_returns) - horizon + 1):
            window_returns = test_returns[i:i+horizon]
            
            mse_return = np.mean(window_returns**2)
            
            h_window = h
            for j, ret in enumerate(test_returns[i-1:i+horizon-1]):  
                h_window = omega + alpha * ret**2 + beta * h_window if j < len(test_returns[i-1:i+horizon-1]) else omega + (alpha + beta) * h_window
            
            sigma_forecast = np.sqrt(h_window)
            realized_vol = np.std(window_returns)
            
            squared_errors.append((sigma_forecast - realized_vol)**2)
        
        mse_by_horizon[f'h_{horizon}'] = {
            'mse_return': float(np.mean([np.mean(test_returns[i:i+horizon]**2) for i in range(len(test_returns) - horizon + 1)])),
            'mse_volatility': float(np.mean(squared_errors)),  
        }
    
    return mse_by_horizon
