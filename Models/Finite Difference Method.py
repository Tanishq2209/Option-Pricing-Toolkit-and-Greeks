import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FiniteDifferenceModel:
    # Finite difference method for European option pricing (Black-Scholes PDE)
    # Supports explicit, implicit and Crank-Nicolson schemes
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100,
                 a=-2.0, b=2.0, m=200, nu_max=2000):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.a = a
        self.b = b
        self.m = m
        self.nu_max = nu_max

    #  Thomas algorithm                                                   
    @staticmethod
    def _thomas_solve(main, upper, lower, rhs):
        n = len(main)
        c = np.zeros(n - 1)
        d = np.zeros(n)
        x = np.zeros(n)

        c[0] = upper[0] / main[0]
        d[0] = rhs[0] / main[0]

        for i in range(1, n):
            m = main[i] - lower[i - 1] * c[i - 1]
            if i < n - 1:
                c[i] = upper[i] / m
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m

        x[-1] = d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    #  Explicit scheme                                                    
    def price_explicit(self, option='call', S0=None, r=None, sigma=None,
                       T=None, K=None, a=None, b=None, m=None, nu_max=None):
        S0 = S0 if S0 is not None else self.S0
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma
        T = T if T is not None else self.T
        K = K if K is not None else self.K
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        m = m if m is not None else self.m
        nu_max = nu_max if nu_max is not None else self.nu_max

        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")

        q = (2 * r) / sigma ** 2
        delta_x = (b - a) / m
        delta_t = (T * sigma ** 2) / (2 * nu_max)
        fidi_lambda = delta_t / delta_x ** 2

        x = np.linspace(a, b, m + 1)
        t = np.linspace(0, nu_max * delta_t, nu_max + 1)

        w = np.zeros((nu_max + 1, m + 1))
        if option == 'call':
            w[0] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)
        else:
            w[0] = np.maximum(np.exp(0.5 * x * (q - 1)) - np.exp(0.5 * x * (q + 1)), 0)

        d = np.zeros(m - 1)
        A = (np.diag(np.repeat(fidi_lambda, m - 2), -1)
             + np.diag(np.repeat(1 - 2 * fidi_lambda, m - 1))
             + np.diag(np.repeat(fidi_lambda, m - 2), 1))

        for nu in range(nu_max):
            tau = T - (2 / sigma ** 2) * t[nu]
            S_right = K * np.exp(x[-1])
            if option == 'call':
                v_right = S_right - K * np.exp(-r * (T - tau))
            else:
                v_right = 0.0
            w_right = (v_right / K) * np.exp(0.5 * (q - 1) * x[-1]
                       + 0.5 * sigma ** 2 * (T - tau) * ((q - 1) ** 2 / 4 + q))
            d[-1] = fidi_lambda * w_right
            w[nu + 1, 1:-1] = A @ w[nu, 1:-1] + d

        S = K * np.exp(x[1:-1])
        t_orig = T - (2 / sigma ** 2) * t
        t_orig = t_orig[::-1]
        V = K * w[::-1, 1:-1] * np.exp(-x[1:-1] / 2 * (q - 1)
             - sigma ** 2 / 2 * (T - t_orig[:, np.newaxis])
             * ((q - 1) ** 2 / 4 + q))
        return t_orig, S, V

    #  Implicit scheme                                                    
    def price_implicit(self, option='call', S0=None, r=None, sigma=None,
                       T=None, K=None, a=None, b=None, m=None, nu_max=None):
        S0 = S0 if S0 is not None else self.S0
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma
        T = T if T is not None else self.T
        K = K if K is not None else self.K
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        m = m if m is not None else self.m
        nu_max = nu_max if nu_max is not None else self.nu_max

        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")

        q = (2 * r) / sigma ** 2
        delta_x = (b - a) / m
        delta_t = (T * sigma ** 2) / (2 * nu_max)
        fidi_lambda = delta_t / delta_x ** 2

        x = np.linspace(a, b, m + 1)
        t = np.linspace(0, nu_max * delta_t, nu_max + 1)

        w = np.zeros((nu_max + 1, m + 1))
        if option == 'call':
            w[0] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)
        else:
            w[0] = np.maximum(np.exp(0.5 * x * (q - 1)) - np.exp(0.5 * x * (q + 1)), 0)

        n = m - 1
        main = np.repeat(1 + 2 * fidi_lambda, n)
        off = np.repeat(-fidi_lambda, n - 1)
        A = sp.diags([off, main, off], [-1, 0, 1], format='csc')
        A_lu = spla.splu(A)

        d = np.zeros(n)
        for nu in range(nu_max):
            tau = T - (2 / sigma ** 2) * t[nu]
            S_right = K * np.exp(x[-1])
            if option == 'call':
                v_right = S_right - K * np.exp(-r * (T - tau))
            else:
                v_right = 0.0
            w_right = (v_right / K) * np.exp(0.5 * (q - 1) * x[-1]
                       + 0.5 * sigma ** 2 * (T - tau) * ((q - 1) ** 2 / 4 + q))
            d[:] = 0.0
            d[-1] = fidi_lambda * w_right
            rhs = w[nu, 1:-1] + d
            w[nu + 1, 1:-1] = A_lu.solve(rhs)

        S = K * np.exp(x[1:-1])
        t_orig = T - (2 / sigma ** 2) * t
        t_orig = t_orig[::-1]
        V = K * w[::-1, 1:-1] * np.exp(-x[1:-1] / 2 * (q - 1)
             - sigma ** 2 / 2 * (T - t_orig[:, np.newaxis])
             * ((q - 1) ** 2 / 4 + q))
        return t_orig, S, V

    #  Crank-Nicolson scheme                                              
    def price_crank_nicolson(self, option='call', S0=None, r=None, sigma=None,
                             T=None, K=None, a=None, b=None, m=None, nu_max=None):
        S0 = S0 if S0 is not None else self.S0
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma
        T = T if T is not None else self.T
        K = K if K is not None else self.K
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        m = m if m is not None else self.m
        nu_max = nu_max if nu_max is not None else self.nu_max

        if option not in ['call', 'put']:
            raise ValueError("option must be 'call' or 'put'")

        q = (2 * r) / (sigma ** 2)
        delta_x = (b - a) / m
        delta_t = (T * sigma ** 2) / (2 * nu_max)
        fidi_lambda = delta_t / delta_x ** 2

        x = np.linspace(a, b, m + 1)
        t = np.linspace(0, delta_t * nu_max, nu_max + 1)

        # Boundary term r2 depends on option type (call vs put swap)
        if option == 'call':
            def r2(t, x):
                return (np.exp(0.5 * (q + 1) * x + 0.25 * (q + 1) ** 2 * t)
                        - np.exp(0.5 * (q - 1) * x + 0.25 * (q - 1) ** 2 * t))
        else:
            def r2(t, x):
                return (np.exp(0.5 * (q - 1) * x + 0.25 * (q - 1) ** 2 * t)
                        - np.exp(0.5 * (q + 1) * x + 0.25 * (q + 1) ** 2 * t))

        # Initial condition
        w = np.zeros((nu_max + 1, m + 1))
        if option == 'call':
            w[0] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0)
        else:
            w[0] = np.maximum(np.exp(0.5 * x * (q - 1)) - np.exp(0.5 * x * (q + 1)), 0)

        # Crank-Nicolson matrices:  A w^{nu+1} = B w^{nu} + d
        n = m - 1
        alpha_A = np.full(n, 1 + fidi_lambda)          # main diag of A
        beta_A = np.full(n - 1, -fidi_lambda / 2)      # upper diag of A
        gamma_A = np.full(n - 1, -fidi_lambda / 2)     # lower diag of A

        alpha_B = np.full(n, 1 - fidi_lambda)          # main diag of B
        beta_B = np.full(n - 1, fidi_lambda / 2)       # upper diag of B
        gamma_B = np.full(n - 1, fidi_lambda / 2)      # lower diag of B
        B = np.diag(alpha_B) + np.diag(beta_B, 1) + np.diag(gamma_B, -1)

        for nu in range(1, nu_max + 1):
            d = np.zeros(n)
            # r1 = 0 at left boundary, so only right boundary contributes
            d[-1] = 0.5 * fidi_lambda * (r2(t[nu], b) + r2(t[nu - 1], b))
            rhs = B @ w[nu - 1, 1:-1] + d
            w[nu, 1:-1] = self._thomas_solve(alpha_A, beta_A, gamma_A, rhs)

        S = K * np.exp(x[1:-1])
        t_orig = T - (2 / sigma ** 2) * t
        t_orig = t_orig[::-1]
        V = K * w[::-1, 1:-1] * np.exp(-x[1:-1] / 2 * (q - 1)
             - sigma ** 2 / 2 * (T - t_orig[:, np.newaxis])
             * ((q - 1) ** 2 / 4 + q))
        return t_orig, S, V


    def price(self, scheme='explicit', **kwargs):
        if scheme == 'explicit':
            return self.price_explicit(**kwargs)
        elif scheme == 'implicit':
            return self.price_implicit(**kwargs)
        elif scheme in ('cn', 'crank_nicolson', 'crank-nicolson'):
            return self.price_crank_nicolson(**kwargs)
        else:
            raise ValueError("scheme must be 'explicit', 'implicit', or 'cn'")


    #  Convenience: price at S0                                           
    def price_at_S0(self, scheme='explicit', option='call', S0=None, **kwargs):
        S0 = S0 if S0 is not None else self.S0
        t, S, V = self.price(scheme=scheme, option=option, S0=S0, **kwargs)
        idx = np.argmin(np.abs(S - S0))
        return V[-1, idx]

    #  Greeks 
    def delta(self, scheme='explicit', option='call', h=None, **kwargs):
        S0 = kwargs.get('S0', self.S0)
        h = h if h is not None else 0.01 * S0
        up = self.price_at_S0(scheme, option, S0=S0 + h, **kwargs)
        dn = self.price_at_S0(scheme, option, S0=S0 - h, **kwargs)
        return (up - dn) / (2 * h)

    def gamma(self, scheme='explicit', option='call', h=None, **kwargs):
        S0 = kwargs.get('S0', self.S0)
        h = h if h is not None else 0.01 * S0
        up = self.price_at_S0(scheme, option, S0=S0 + h, **kwargs)
        mid = self.price_at_S0(scheme, option, S0=S0, **kwargs)
        dn = self.price_at_S0(scheme, option, S0=S0 - h, **kwargs)
        return (up - 2 * mid + dn) / (h ** 2)

    def vega(self, scheme='explicit', option='call', h=None, **kwargs):
        sigma = kwargs.get('sigma', self.sigma)
        h = h if h is not None else 0.01 * sigma
        up = self.price_at_S0(scheme, option, sigma=sigma + h, **kwargs)
        dn = self.price_at_S0(scheme, option, sigma=sigma - h, **kwargs)
        return (up - dn) / (2 * h)

    def theta(self, scheme='explicit', option='call', h=None, **kwargs):
        T = kwargs.get('T', self.T)
        h = h if h is not None else 0.01 * T
        up = self.price_at_S0(scheme, option, T=T + h, **kwargs)
        dn = self.price_at_S0(scheme, option, T=T - h, **kwargs)
        return -(up - dn) / (2 * h)

    def rho(self, scheme='explicit', option='call', h=None, **kwargs):
        r = kwargs.get('r', self.r)
        h = h if h is not None else 0.0001
        up = self.price_at_S0(scheme, option, r=r + h, **kwargs)
        dn = self.price_at_S0(scheme, option, r=r - h, **kwargs)
        return (up - dn) / (2 * h)