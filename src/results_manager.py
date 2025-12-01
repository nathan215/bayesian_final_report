import pickle
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

class ResultsManager:
    
    def __init__(self, base_dir='../results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.ar_dir = self.base_dir / 'ar'
        self.garch_dir = self.base_dir / 'garch'
        self.hierarchical_dir = self.base_dir / 'hierarchical'
        self.comparison_dir = self.base_dir / 'comparison'
        
        for d in [self.ar_dir, self.garch_dir, self.hierarchical_dir, self.comparison_dir]:
            d.mkdir(exist_ok=True)
    
    def _validate_is_dict(self, data: Any, name: str):
        """Ensures data is a dictionary. Raises TypeError if not."""
        if not isinstance(data, dict):
            raise TypeError(
                f"Error saving {name}: Expected dict, got {type(data).__name__}. "
                "If you passed a set like {a, b}, convert it to a dictionary like {'key': a, 'key2': b}."
            )

    def save_ar(self, ticker: str, method: str, results: Dict[str, Any], prior_set: str = None):
        """Save Autoregressive (AR) model results."""
        self._validate_is_dict(results, f"AR {method} results")
        
        if method == 'bayes' and prior_set is None:
            raise ValueError("prior_set is REQUIRED when method='bayes' for AR results.")
        
        filename = f'{ticker}_ar_{method}'
        if prior_set:
            filename += f'_{prior_set}'
        filename += '.pkl'
        
        path = self.ar_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved AR: {path.name}")
        return path
    
    def load_ar(self, ticker: str, method: str, prior_set: str = None) -> Dict[str, Any]:
        """Load Autoregressive (AR) model results."""
        if method == 'bayes' and prior_set is None:
            raise ValueError("prior_set is REQUIRED when method='bayes' for AR results.")
        
        filename = f'{ticker}_ar_{method}'
        if prior_set:
            filename += f'_{prior_set}'
        filename += '.pkl'
        
        path = self.ar_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"AR result file not found: {path.name}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def list_ar(self, ticker: str = None) -> pd.DataFrame:
        """List all AR results with MSE for horizons h=1, 5, 22."""
        files = list(self.ar_dir.glob('*.pkl'))
        if ticker:
            files = [f for f in files if f.stem.startswith(ticker)]
        
        results_list = []
        for f in sorted(files):
            try:
                with open(f, 'rb') as fp:
                    res = pickle.load(fp)
            except Exception as e:
                print(f"Warning: Could not load {f.name}. Skipping. Error: {e}")
                continue

            parts = f.stem.split('_')
            current_ticker = parts[0]
            current_method = parts[2] #
            current_prior = parts[3] if len(parts) > 3 else 'N/A' 
            
            mse_h1 = np.nan
            mse_h5 = np.nan
            mse_h22 = np.nan
            
            if 'forecasts' in res:
                mse_h1 = res['forecasts'].get('h_1', {}).get('mse', np.nan)
                mse_h5 = res['forecasts'].get('h_5', {}).get('mse', np.nan)
                mse_h22 = res['forecasts'].get('h_22', {}).get('mse', np.nan)
            
            runtime = res.get('runtime', 0.0) 
            
            results_list.append({
                'File': f.name,
                'Ticker': current_ticker,
                'Method': current_method,
                'Prior': current_prior,
                'MSE(h=1)': mse_h1,
                'MSE(h=5)': mse_h5,
                'MSE(h=22)': mse_h22,
                'Runtime(s)': runtime
            })
        
        return pd.DataFrame(results_list)
    
    def save_garch(self, ticker: str, method: str, results: Dict[str, Any], prior_set: str = None, diagnostics: Dict[str, Any] = None):
        """Save GARCH model results."""
        self._validate_is_dict(results, f"GARCH {method} results")
        
        if method == 'bayes' and prior_set is None:
            raise ValueError("prior_set is REQUIRED when method='bayes' for GARCH results.")
        
        filename = f'{ticker}_garch_{method}'
        if prior_set:
            filename += f'_{prior_set}'
        filename += '.pkl'
        
        path = self.garch_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        
        if diagnostics is not None:
            diag_path = path.with_suffix('.json')
            diag_clean = self._numpy_to_native(diagnostics) 
            with open(diag_path, 'w') as f:
                json.dump(diag_clean, f, indent=2)
        
        print(f"Saved GARCH: {path.name}")
        return path
    
    def load_garch(self, ticker: str, method: str, prior_set: str = None) -> Dict[str, Any]:
        """Load GARCH model results."""
        if method == 'bayes' and prior_set is None:
            raise ValueError("prior_set is REQUIRED when method='bayes' for GARCH results.")
        
        filename = f'{ticker}_garch_{method}'
        if prior_set:
            filename += f'_{prior_set}'
        filename += '.pkl'
        
        path = self.garch_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"GARCH result file not found: {path.name}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def list_garch(self, ticker: str = None) -> pd.DataFrame:
        """List all GARCH results with MSE for horizons h=1, 5, 22."""
        files = list(self.garch_dir.glob('*.pkl'))
        if ticker:
            files = [f for f in files if f.stem.startswith(ticker)]
        
        results_list = []
        for f in sorted(files):
            try:
                with open(f, 'rb') as fp:
                    res = pickle.load(fp)
            except Exception as e:
                print(f"Warning: Could not load {f.name}. Skipping. Error: {e}")
                continue

            parts = f.stem.split('_')
            current_ticker = parts[0]
            current_method = parts[2]
            current_prior = parts[3] if len(parts) > 3 else 'N/A'
            
            mse_h1 = np.nan
            mse_h5 = np.nan
            mse_h22 = np.nan
            
            if 'forecasts' in res:
                mse_h1 = res['forecasts'].get('h_1', {}).get('mse', np.nan)
                mse_h5 = res['forecasts'].get('h_5', {}).get('mse', np.nan)
                mse_h22 = res['forecasts'].get('h_22', {}).get('mse', np.nan)
            
            runtime = res.get('runtime', 0.0)
            
            results_list.append({
                'File': f.name,
                'Ticker': current_ticker,
                'Method': current_method,
                'Prior': current_prior,
                'MSE(h=1)': mse_h1,
                'MSE(h=5)': mse_h5,
                'MSE(h=22)': mse_h22,
                'Runtime(s)': runtime
            })
        
        return pd.DataFrame(results_list)
    
    def save_hierarchical(self, model_type: str, results: Dict[str, Any], diagnostics: Dict[str, Any] = None):
        """
        Save hierarchical model results.
        model_type: Specify 'ar' for Hierarchical AR or 'garch' for Hierarchical GARCH.
        """
        self._validate_is_dict(results, f"Hierarchical {model_type} results")
        
        if model_type not in ['ar', 'garch']:
            raise ValueError("model_type must be 'ar' or 'garch' for hierarchical results.")

        filename = f'hierarchical_{model_type}.pkl'
        path = self.hierarchical_dir / filename
        
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        
        if diagnostics is not None:
            diag_path = path.with_suffix('.json')
            diag_clean = self._numpy_to_native(diagnostics)
            with open(diag_path, 'w') as f:
                json.dump(diag_clean, f, indent=2)
        
        print(f"Saved Hierarchical {model_type.upper()}: {path.name}")
        return path
    
    def load_hierarchical(self, model_type: str) -> Dict[str, Any]:
        """
        Load hierarchical model results.
        model_type: Specify 'ar' for Hierarchical AR or 'garch' for Hierarchical GARCH.
        """
        if model_type not in ['ar', 'garch']:
            raise ValueError("model_type must be 'ar' or 'garch' for hierarchical results.")

        filename = f'hierarchical_{model_type}.pkl'
        path = self.hierarchical_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Hierarchical {model_type.upper()} result file not found: {path.name}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def list_hierarchical(self) -> pd.DataFrame:
        """List all hierarchical results."""
        files = list(self.hierarchical_dir.glob('*.pkl'))
        
        results_list = []
        for f in sorted(files):
            try:
                with open(f, 'rb') as fp:
                    res = pickle.load(fp)
            except Exception as e:
                print(f"Warning: Could not load {f.name}. Skipping. Error: {e}")
                continue

            model_type = f.stem.replace('hierarchical_', '')
            runtime = res.get('runtime', np.nan)
            
            results_list.append({
                'File': f.name,
                'Model Type': model_type.upper(),
                'Runtime(s)': runtime
            })
        
        return pd.DataFrame(results_list)

    def save_comparison(self, comparison_name: str, comparison_table: pd.DataFrame):
        """Save comparison table as CSV."""
        path = self.comparison_dir / f'{comparison_name}.csv'
        comparison_table.to_csv(path, index=False) 
        print(f"Saved Comparison: {path.name}")
        return path
    
    def load_comparison(self, comparison_name: str) -> pd.DataFrame:
        """Load comparison table from CSV."""
        path = self.comparison_dir / f'{comparison_name}.csv'
        if not path.exists():
            raise FileNotFoundError(f"Comparison file not found: {path.name}")
        return pd.read_csv(path) 
    
    def list_comparison(self) -> pd.DataFrame:
        """List all comparison tables."""
        files = list(self.comparison_dir.glob('*.csv'))
        
        results_list = []
        for f in sorted(files):
            results_list.append({'File': f.name, 'Name': f.stem})
        
        return pd.DataFrame(results_list)

    @staticmethod
    def _numpy_to_native(obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: ResultsManager._numpy_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultsManager._numpy_to_native(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)): 
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        else:
            return obj



