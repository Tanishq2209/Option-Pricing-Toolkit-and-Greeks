import numpy as np
from scipy.stats import norm

class MonteCarloModel:
    # Monte Carlo simulation for European option pricing.
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100, n_simulations=10000, seed=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.n_simulations = n_simulations
        self.seed = seed
        
        # Random number generator
        self._rng = np.random.default_rng(seed)
    
    def _generate_paths(self, S0, r, sigma, T, n_simulations):
        Z = self._rng.normal(0, 1, n_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return ST
    
    def price(self, option='call', t=0, St=None, K=None, T=None, r=None, sigma=None, n_simulations=None, return_ci=True):
        # Use instance defaults if not provided
        St = St if St is not None else self.S0
        K = K if K is not None else self.K
        T = T if T is not None else self.T
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma
        n_simulations = n_simulations if n_simulations is not None else self.n_simulations
        
        # Input validation
        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")
        if any(x <= 0 for x in [St, K, T, sigma, n_simulations]):
            raise ValueError("St, K, T, sigma, and n_simulations must be positive")
        if T <= t:
            raise ValueError("T must be greater than t")
        
        # Generate terminal stock prices
        # For simplicity, we use time to maturity T - t
        ST = self._generate_paths(St, r, sigma, T - t, n_simulations)
        
        # Compute payoffs
        if option == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
        
        # Discount and average
        discount_factor = np.exp(-r * (T - t))
        price_estimate = discount_factor * np.mean(payoffs)
        
        if return_ci:
            # Standard error and 95% confidence interval
            std_error = discount_factor * np.std(payoffs) / np.sqrt(n_simulations)
            ci_lower = price_estimate - 1.96 * std_error
            ci_upper = price_estimate + 1.96 * std_error
            return price_estimate, std_error, ci_lower, ci_upper
        else:
            return price_estimate