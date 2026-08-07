import numpy as np

class CRRModel:
    # CRR model supports both European / American Options
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100, M=100):
             self.S0 = S0
             self.r = r 
             self.sigma = sigma 
             self.T = T
             self.K = K
             self.M = M

            # Compute constants for further analysis
            self.delta_t = T / M
            self.beta = (np.exp(-r * self.delta_t)+np.exp((r+(sigma)**2) * self.delta_t)) / 2
            self.u = self.beta + np.sqrt(self.beta**2 - 1)
            self.d = 1 / self.u
            self.q = (np.exp(r * self.delta_t) - self.d) / (self.u - self.d)

    def Stock_tree(self):
        S = np.empty((self.M + 1, self.M + 1))
        for i in range(self.M + 1):
            for j in range(i + 1):
                S[j, i] = self.S0 * (self.u**j) * (self.d**(i-j))
        return S
  
    def price(self, S0=None, r=None, sigma=None, T=None, M=None, K=None, option_type='call', option_style='European'):
        # Implement if parameters not provided
        S0 = S0 if S0 is not None else self.S0
        K = K if K is not None else self.K
        r = r if r is not None else self.r
        T = T if T is not None else self.T
        sigma = sigma if sigma is not None else self.sigma
        M = M if M is not None else self.M
        
        S = self.Stock_tree()
        V = np.zeros((self.M + 1, self.M + 1))
        for j in range(M + 1):
            # Terminal Payoff
            if option_type == 'call':
                V[j, self.M] = max(0, S[j, self.M] - K)
            else:   # put
                V[j, self.M] = max(0, K - S[j, self.M])
        
        # Backward induction
        for i in range(M - 1, -1, -1):
            for j in range(i + 1):
                if option_style == 'European':
                    V[j, i] = np.exp(-r * delta_t) * (q * V[j+1, i+1] + (1-q) * V[j, i+1])
                else:  # American
                    # Exercise value
                    if option_type == 'call':
                        exercise = max(0, S[j, i] - K)
                    else:
                        exercise = max(0, K - S[j, i])
                    hold = np.exp(-r * delta_t) * (q * V[j+1, i+1] + (1-q) * V[j, i+1])
                    V[j, i] = max(exercise, hold)
        
        return V[0, 0]
