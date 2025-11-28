import numpy as np

class PriorSpecification:
    def __init__(self, model_type):
        self.model_type = model_type
        self.priors = {}
    
    def display(self):
        print(f"PRIOR SPECIFICATION: {self.model_type.upper()}")     
        for param, spec in self.priors.items():
            print(f"\n{param}:")
            for key, value in spec.items():
                print(f"  {key}: {value}")

class AR1_Priors(PriorSpecification):
    """
    AR(1) Model: r_t = μ + φ*r_{t-1} + ε_t
    
    PRIOR CHOICES:
    - μ: Normal(0, 0.01) - daily returns typically -0.5% to +0.5%, mean near 0
    - φ: Normal(0.1, 0.3) - weak autocorrelation expected
    - σ: HalfNormal(0.1) - WEAKLY INFORMATIVE scale prior (not InverseGamma)
    """
    
    def __init__(self):
        super().__init__('AR(1)')
        
        self.priors = {
            'mu': {
                'distribution': 'Normal',
                'mu': 0.0,           
                'sigma': 0.01,        
                'interpretation': 'Mean return is very small; daily returns typically ±0.5%'
            },
            
            'phi': {
                'distribution': 'Normal',
                'mu': 0.1,            
                'sigma': 0.3,         
                'lower_bound': -0.99, 
                'upper_bound': 0.99,  
                'interpretation': 'Weak autocorrelation expected; allow both directions'
            },            

            'sigma_sq': {
                'distribution': 'HalfCauchy',
                'beta': 1.0,        
                'interpretation': 'VERY WEAKLY INFORMATIVE - allows large volatility spikes'
            }
        }
    
    def get_pymc_spec(self):
        return {
            'mu': ('Normal', {'mu': 0, 'sigma': 0.01}),
            'phi': ('Normal', {'mu': 0.1, 'sigma': 0.3}),
            'sigma_sq': ('HalfCauchy', {'beta': 1.0})
        }


class GARCH_Priors(PriorSpecification):
    """
    GARCH(1,1) Model: σ_t² = ω + α*ε_{t-1}² + β*σ_{t-1}²
    """
    
    def __init__(self):
        super().__init__('GARCH(1,1)')
        
        self.priors = {
            'mu': {
                'distribution': 'Normal',
                'mu': 0.0,
                'sigma': 0.01,
                'interpretation': 'Mean return (same as AR(1))'
            },
            
            'omega': {
                'distribution': 'HalfNormal',
                'sigma': 0.05,      
                'interpretation': 'Baseline volatility. HalfNormal allows flexibility'
            },
            
            'alpha': {
                'distribution': 'Beta',
                'alpha': 2.0,        
                'beta': 5.0,        
                'mean': 0.286,      
                'mode': 0.143,      
                'range': [0, 1],     
                'interpretation': 'Shock effect on volatility. Temporary but immediate impact'
            },
            
            'beta': {
                'distribution': 'Beta',
                'alpha': 5.0,        
                'beta': 2.0,        
                'mean': 0.714,       
                'mode': 0.667,       
                'range': [0, 1],     
                'interpretation': 'Volatility persistence. High persistence expected'
            },
            
            'constraint': {
                'name': 'stationarity',
                'condition': 'alpha + beta < 1',
                'typical_range': '0.90-0.99',
                'interpretation': 'Critical for model stability and stationarity'
            }
        }
    
    def get_pymc_spec(self):
        """Return specification formatted for PyMC"""
        return {
            'mu': ('Normal', {'mu': 0, 'sigma': 0.01}),
            'omega': ('HalfNormal', {'sigma': 0.05}), 
            'alpha': ('Beta', {'alpha': 2, 'beta': 5}),
            'beta': ('Beta', {'alpha': 5, 'beta': 2})
        }

class SV_Priors(PriorSpecification):
    """
    SV Model: r_t = μ + exp(h_t/2)*ε_t
               h_t = α + φ*h_{t-1} + η_t
    """
    
    def __init__(self):
        super().__init__('SV')
        
        self.priors = {
            'mu': {
                'distribution': 'Normal',
                'mu': 0.0,
                'sigma': 0.01,
                'interpretation': 'Mean return (same as before)'
            },
            
            'alpha': {
                'distribution': 'Normal',
                'mu': -2.0,           
                'sigma': 1.0,        
                'interpretation': 'Baseline log-volatility level. Mean ~37% volatility range'
            },
            
            'phi': {
                'distribution': 'Normal',
                'mu': 0.95,         
                'sigma': 0.05,       
                'lower_bound': -0.99, 
                'upper_bound': 0.99,  
                'typical_range': '0.90-0.99',
                'interpretation': 'Log-volatility persistence. Expected very high'
            },
            
            'sigma_eta': {
                'distribution': 'HalfNormal',  # ← Changed from InverseGamma
                'sigma': 0.5,                  # ← More flexible scale
                'interpretation': 'Volatility shock scale. Controls day-to-day vol changes'
            },
            
            'nu': {
                'distribution': 'Gamma', 
                'alpha': 2.0,
                'beta': 0.1,
                'mean': 20.0,       
                'interpretation': 'Degrees of freedom for Student-t errors. Your data kurtosis: 3-26'
            }
        }
    
    def get_pymc_spec(self, error_dist='normal'):
        spec = {
            'mu': ('Normal', {'mu': 0, 'sigma': 0.01}),
            'alpha': ('Normal', {'mu': -2, 'sigma': 1}),
            'phi': ('Normal', {'mu': 0.95, 'sigma': 0.05}),
            'sigma_eta': ('HalfNormal', {'sigma': 0.5})
        }
        
        if error_dist == 'student_t':
            spec['nu'] = ('Gamma', {'alpha': 2, 'beta': 0.1})
        
        return spec


AR1_priors = AR1_Priors()
GARCH_priors = GARCH_Priors()
SV_priors = SV_Priors()


def display_all_priors():
    AR1_priors.display()
    print("\n" + "="*80 + "\n")
    GARCH_priors.display()
    print("\n" + "="*80 + "\n")
    SV_priors.display()


def adjust_priors_for_stock(stock_name, returns_data):
    """
    Optional: Adjust priors based on observed data statistics
    """
    obs_mean = np.mean(returns_data)
    obs_std = np.std(returns_data)
    
    return {
        'ar1': AR1_priors.get_pymc_spec(),
        'garch': GARCH_priors.get_pymc_spec(),
        'sv': SV_priors.get_pymc_spec()
    }
