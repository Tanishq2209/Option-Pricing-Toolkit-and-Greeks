import numpy as np
from scipy.integrate import quad

class LaplaceHestonModel:
    # Laplace transform method for European option pricing in Heston model
    def __init__(self, S0=100, r=0.05, T=1, K=100, gam0=0.04, kappa=1.0,
                 lamb=0.0, sig_tilde=0.2, rho=-0.7, R=1.5, upper=100, h=1e-4):
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K
        self.gam0 = gam0
        self.kappa = kappa
        self.lamb = lamb
        self.sig_tilde = sig_tilde
        self.rho = rho
        self.R = R
        self.upper = upper
        self.h = h

    def _heston_char(self, u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho):
        B = rho * sig_tilde * u * 1j - lamb
        D = np.sqrt(B**2 + sig_tilde**2 * u * (u + 1j))
        lnG = np.log(B * (np.exp(-D * (T - t)) - 1) / (2 * D) + (np.exp(-D * (T - t)) + 1) / 2)
        psi_0 = u * 1j * r * (T - t) - kappa / sig_tilde**2 * ((B + D) * (T - t) + 2 * lnG)
        psi_1 = u * 1j
        psi_2 = u * (u + 1j) * (np.exp(-D * (T - t)) - 1) / (B * (np.exp(-D * (T - t)) - 1) + D * (np.exp(-D * (T - t)) + 1))
        return np.exp(psi_0 + psi_1 * np.log(St) + psi_2 * gamt)

    def _f_tilde(self, z, K):
        return (K ** (1 - z)) / (z * (z - 1))

    def _integrand(self, u, K, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho, R):
        ft = self._f_tilde(R + 1j * u, K)
        char = self._heston_char(u - 1j * R, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho)
        return np.real(ft * char)

    def price(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, upper=None):
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
        upper = upper if upper is not None else self.upper

        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")

        integral = quad(
            lambda u: self._integrand(u, K, St, t, T, r, gam0, kappa, lamb,
                                      sig_tilde, rho, R),
            0, upper
        )[0]
        call_price = np.exp(-r * (T - t)) / np.pi * integral

        if option == 'call':
            return call_price
        else:
            return call_price - St + K * np.exp(-r * (T - t))

    def delta(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, upper=None, h=None):
        St = St if St is not None else self.S0
        h = h if h is not None else self.h
        up = self.price(option, t, St + h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        dn = self.price(option, t, St - h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        return (up - dn) / (2 * h)

    def gamma(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, upper=None, h=None):
        St = St if St is not None else self.S0
        h = h if h is not None else self.h
        up = self.price(option, t, St + h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        mid = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                         sig_tilde, rho, R, upper)
        dn = self.price(option, t, St - h, K, T, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        return (up - 2 * mid + dn) / (h ** 2)

    def vega(self, option='call', t=0, St=None, K=None, T=None, r=None,
             gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
             R=None, upper=None, h=None):
        St = St if St is not None else self.S0
        sig_tilde = sig_tilde if sig_tilde is not None else self.sig_tilde
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                        sig_tilde + h, rho, R, upper)
        dn = self.price(option, t, St, K, T, r, gam0, kappa, lamb,
                        sig_tilde - h, rho, R, upper)
        return (up - dn) / (2 * h)

    def theta(self, option='call', t=0, St=None, K=None, T=None, r=None,
              gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
              R=None, upper=None, h=None):
        St = St if St is not None else self.S0
        T = T if T is not None else self.T
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T + h, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        dn = self.price(option, t, St, K, T - h, r, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        return -(up - dn) / (2 * h)

    def rho(self, option='call', t=0, St=None, K=None, T=None, r=None,
            gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
            R=None, upper=None, h=None):
        St = St if St is not None else self.S0
        r = r if r is not None else self.r
        h = h if h is not None else self.h
        up = self.price(option, t, St, K, T, r + h, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        dn = self.price(option, t, St, K, T, r - h, gam0, kappa, lamb,
                        sig_tilde, rho, R, upper)
        return (up - dn) / (2 * h)
