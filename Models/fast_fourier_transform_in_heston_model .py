import numpy as np

class FFTHestonModel:
    # FFT method for European option pricing in Heston model
    def __init__(self, S0=100, r=0.05, T=1, K=None, gam0=0.04, kappa=1.0,
                 lamb=0.0, sig_tilde=0.2, rho=-0.7, R=1.5, M=5000, N=2**15, h=1e-4):
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K if K is not None else np.linspace(50, 300, 100)
        self.gam0 = gam0
        self.kappa = kappa
        self.lamb = lamb
        self.sig_tilde = sig_tilde
        self.rho = rho
        self.R = R
        self.M = M
        self.N = N
        self.h = h

    def _heston_char(self, u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho):
        B = rho * sig_tilde * u * 1j - lamb
        D = np.sqrt(B**2 + sig_tilde**2 * u * (u + 1j))
        lnG = np.log(B * (np.exp(-D * (T - t)) - 1) / (2 * D) + (np.exp(-D * (T - t)) + 1) / 2)
        psi_0 = u * 1j * r * (T - t) - kappa / sig_tilde**2 * ((B + D) * (T - t) + 2 * lnG)
        psi_1 = u * 1j
        psi_2 = u * (u + 1j) * (np.exp(-D * (T - t)) - 1) / (B * (np.exp(-D * (T - t)) - 1) + D * (np.exp(-D * (T - t)) + 1))
        return np.exp(psi_0 + psi_1 * np.log(St) + psi_2 * gamt)

    def _g(self, u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho, R):
        z = R + 1j * u
        ft = 1 / (z * (z - 1))
        char = self._heston_char(u - 1j * R, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho)
        return ft * char

    def price(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, M=None, N=None):
        St = St if St is not None else self.S0
        K = K if K is not None else self.K
        T = T if T is not None else self.T
        r = r if r is not None else self.r
        gam0 = gam0 if gam0 is not None else self.gam0
        kappa = kappa if kappa is not None else self.kappa
        lamb = lamb if lamb is not None else self.lamb
        sig_tilde = sig_tilde if sig_tilde is not None else self.sig_tilde
        rho = rho if rho is not None else self.rho
        R = R if R is not None else self.R
        M = M if M is not None else self.M
        N = N if N is not None else self.N

        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")

        K = np.asarray(K, dtype=float)
        n = np.arange(1, N + 1, 1)
        k1 = np.log(K[0])
        kappa_m = k1 + (n - 1) * (2 * np.pi) / M
        delta = M / N

        x = self._g((n - 0.5) * delta, St, t, T, r, gam0, kappa, lamb,
                    sig_tilde, rho, R) * delta * np.exp(-1j * (n - 1) * delta * k1)
        x_hat = np.fft.fft(x)

        V_kappa_m = ((np.exp(-r * (T - t) + (1 - R) * kappa_m)) / np.pi) * \
                    np.real(x_hat * np.exp((-1j / 2) * delta * kappa_m))

        call_prices = np.interp(K, np.exp(kappa_m), V_kappa_m)

        if option == 'call':
            return call_prices
        else:
            return call_prices - St + K * np.exp(-r * (T - t))

    def delta(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, M=None, N=None, h=None):
        St = St if St is not None else self.S0
        h = h if h is not None else self.h
        up = self.price(option, t, St + h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        dn = self.price(option, t, St - h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        return (up - dn) / (2 * h)

    def gamma(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, M=None, N=None, h=None):
        St = St if St is not None else self.S0
        h = h if h is not None else self.h
        up = self.price(option, t, St + h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        mid = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                         sig_tilde, rho, R, M, N)
        dn = self.price(option, t, St - h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        return (up - 2 * mid + dn) / (h ** 2)

    def vega(self, option='call', t=0, St=None, K=None, T=None, r=None,
             gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
             R=None, M=None, N=None, h=None):
        St = St if St is not None else self.S0
        sig_tilde = sig_tilde if sig_tilde is not None else self.sig_tilde
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                        sig_tilde + h, rho, R, M, N)
        dn = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                        sig_tilde - h, rho, R, M, N)
        return (up - dn) / (2 * h)

    def theta(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, M=None, N=None, h=None):
        St = St if St is not None else self.S0
        T = T if T is not None else self.T
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T + h, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        dn = self.price(option, t, St, K, T - h, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        return -(up - dn) / (2 * h)

    def rho(self, option='call', t=0, St=None, K=None, T=None, r=None,
            gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
            R=None, M=None, N=None, h=None):
        St = St if St is not None else self.S0
        r = r if r is not None else self.r
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T, r + h, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        dn = self.price(option, t, St, K, T, r - h, gam0, kappa, lamb,
                        sig_tilde, rho, R, M, N)
        return (up - dn) / (2 * h)
