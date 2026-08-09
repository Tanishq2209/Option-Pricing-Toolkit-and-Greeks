import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    # BS model for European option pricing
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100):
         self.S0 = S0
         self.r = r 
         self.sigma = sigma 
         self.T = T
         self.K = K

    def price(self, t=0, St=None, K=None, T=None, r=None, sigma = None, option='call'):
         # Implement if parameters not provided
         St = St if St is not None else self.S0
         K = K if K is not None else self.K
         r = r if r is not None else self.r
         T = T if T is not None else self.T
         sigma = sigma if sigma is not None else self.sigma

         d1 = ((np.log(St / K) + (r + 0.5 * sigma**2) * (T - t))) / (sigma * np.sqrt(T - t))
         d2 = d1 - sigma * np.sqrt(T - t)
    
         if option == 'call':
              price = St * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
         elif option=='put':
              price = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - St * norm.cdf(-d1)
         else:
              raise ValueError("Option type must be either call or put")

     def delta(self, t=0, St=None, K=None, T=None, r=None, sigma = None, option='call'):
              St = St if St is not None else self.S0
              K = K if K is not None else self.K
              r = r if r is not None else self.r
              T = T if T is not None else self.T
              sigma = sigma if sigma is not None else self.sigma

              d1 = (np.log(St/K) + r * (T-t) + 0.5 * (sigma**2)*(T-t)) / (sigma * np.sqrt(T-t))

              if option == 'call':
                    return norm.cdf(d1)
              else:
                    return norm.cdf(d1) - 1 # put option


     def vega(self, t=0, St=None, K=None, T=None, r=None, sigma = None, option='call'):
               St = St if St is not None else self.S0
               K = K if K is not None else self.K
               r = r if r is not None else self.r
               T = T if T is not None else self.T
               sigma = sigma if sigma is not None else self.sigma

               # same for call and put option
               d1 = (np.log(St/K) + r * (T-t) + 0.5 * (sigma**2)*(T-t)) / (sigma * np.sqrt(T-t))
               return St * norm.pdf(d1) * np.sqrt(T-t)

     def theta(self, t=0, St=None, K=None, T=None, r=None, sigma = None, option='call'):
               St = St if St is not None else self.S0
               K = K if K is not None else self.K
               r = r if r is not None else self.r
               T = T if T is not None else self.T
               sigma = sigma if sigma is not None else self.sigma
          
               d1 = (np.log(St/K) + r * (T-t) + 0.5 * (sigma**2)*(T-t)) / (sigma * np.sqrt(T-t))
               d2 = d1 - sigma * np.sqrt(T-t)
          
               if option == 'call':
                    return -(St * norm.pdf(d1) * sigma) / (2 * np.sqrt(T - t)) - r * K * np.exp(-r * (T - t)) * norm.cdf(d2)
               else: # put option
                    return -(St * norm.pdf(d1) * sigma) / (2 * np.sqrt(T - t)) + r * K * np.exp(-r * (T - t)) * norm.cdf(-d2)
          