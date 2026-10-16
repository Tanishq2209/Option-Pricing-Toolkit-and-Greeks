import math
import numpy as np
import pandas as pd

from Models.black_scholes_model import BlackScholesModel
from Models.cox_ross_rubinstein_model import CRRModel
from Models.monte_carlo_simulation import MonteCarloModel
from Models.finite_difference_method import FiniteDifferenceModel
from Models.laplace_transform_heston_model import LaplaceHestonModel
from Models.fast_ffourier_transform_in_heston_model import FFTHestonModel
                                       
stocks_df  = pd.read_csv('stocks.csv')
options_df = pd.read_csv('options_SP500.csv')
dax_df     = pd.read_csv('DAX_sample.csv')

print("stocks.csv shape      :", stocks_df.shape)
print("options_SP500.csv shape:", options_df.shape)
print("DAX_sample.csv shape  :", dax_df.shape)

#  Parameters                                                    
S_0   = 100
r     = 0.05
sigma = 0.2
T     = 1
K     = 105
M     = 100

#  Instantiate model classes                                          
crr          = CRRModel(S0=S_0, r=r, sigma=sigma, T=T, K=K, M=M)
bs           = BlackScholesModel(S0=S_0, r=r, sigma=sigma, T=T, K=K)
mc           = MonteCarloModel(S0=S_0, r=r, sigma=sigma, T=T, K=K, n_simulations=10000, seed=42)
fidi         = FiniteDifferenceModel(S0=S_0, r=r, sigma=sigma, T=T, K=K, m=100, nu_max=1000)

gam0      = math.pow(0.3, 2)
kappa     = math.pow(0.3, 2)
lamb      = 2.5
sig_tilde = 0.2
rho       = -0.7
R         = 1.5

laplace_heston = LaplaceHestonModel(S0=S_0, r=r, T=T, K=K, gam0=gam0,
                                    kappa=kappa, lamb=lamb,
                                    sig_tilde=sig_tilde, rho=rho, R=R)

fft_heston     = FFTHestonModel(S0=S_0, r=r, T=T, K=np.arange(80, 181),
                                gam0=gam0, kappa=kappa, lamb=lamb,
                                sig_tilde=sig_tilde, rho=rho, R=R,
                                M=5000, N=2**15)


#  CRR Model                                                          
crr_eu_call = crr.price(option_type='call', option_style='European')
print(f"European Call Option Price (CRR model): {crr_eu_call}")

crr_am_put = crr.price(option_type='put', option_style='American')
print(f"American Put Option Price (CRR model): {crr_am_put}")

#  Black-Scholes Model                                                
bs_eu_call = bs.price(t=0, St=S_0, K=K, T=T, r=r, sigma=sigma, option='call')
print(f"European Call Option Price (Black-Scholes model): {bs_eu_call}")

bs_eu_put = bs.price(t=0, St=S_0, K=K, T=T, r=r, sigma=sigma, option='put')
print(f"European Put Option Price (Black-Scholes model): {bs_eu_put}")

bs_greeks = bs.Greeks(t=0, St=S_0, K=K, T=T, r=r, sigma=sigma, option='call')
print(f"Black-Scholes Greeks (Call): {bs_greeks}")


#  Monte Carlo Simulation                                            
mc_eu_call, mc_call_se, mc_call_ci = mc.price(option='call', t=0, St=S_0, K=K, T=T,
                                              r=r, sigma=sigma, n_simulations=10000)
print(f"European Call Option Price (Monte Carlo simulation): {mc_eu_call}")

mc_eu_put, mc_put_se, mc_put_ci = mc.price(option='put', t=0, St=S_0, K=K, T=T,
                                           r=r, sigma=sigma, n_simulations=10000)
print(f"European Put Option Price (Monte Carlo simulation): {mc_eu_put}")

#  Finite Difference Method                                           
fidi_eu_call = fidi.price(scheme='explicit', option='call')
t_grid, S_grid, V_grid = fidi_eu_call
idx = np.argmin(np.abs(S_grid - S_0))
print(f"European Call Option Price (Finite Difference Scheme): {V_grid[-1, idx]}")

fidi_eu_put = fidi.price(scheme='explicit', option='put')
t_grid, S_grid, V_grid = fidi_eu_put
idx = np.argmin(np.abs(S_grid - S_0))
print(f"European Put Option Price (Finite Difference Scheme): {V_grid[-1, idx]}")


#  Laplace Transform Method (Heston Model)                            
heston_put_laplace = laplace_heston.price(option='put', t=0, St=S_0, K=K, T=T, r=r)
print(f"Heston European Put Option Price (Laplace transform method): {heston_put_laplace}")

heston_call_laplace = laplace_heston.price(option='call', t=0, St=S_0, K=K, T=T, r=r)
print(f"Heston European Call Option Price (Laplace transform method): {heston_call_laplace}")


#  Fast Fourier Transform Method (Heston Model)                       
heston_call_fft = fft_heston.price(option='call', t=0, St=S_0, T=T, r=r)
print(f"Heston European Call Option Price (FFT method): {heston_call_fft[0]}")

heston_put_fft = fft_heston.price(option='put', t=0, St=S_0, T=T, r=r)
print(f"Heston European Put Option Price (FFT method): {heston_put_fft[0]}")
