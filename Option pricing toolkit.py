import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq

try:
    import yfinance as yf
except ImportError:
    yf = None


class OptionPricing:
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100, M=50,
                 gam0=0.04, kappa=1.0, lamb=0.0, sig_tilde=0.2, rho=-0.7):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M
        self.gam0 = gam0
        self.kappa = kappa
        self.lamb = lamb
        self.sig_tilde = sig_tilde
        self.rho = rho

    def _validate(self, S0, K, T, sigma, option):
        if option not in ('call', 'put'):
            raise ValueError("option must be 'call' or 'put'")
        if S0 is None or S0 <= 0:
            raise ValueError("S0 must be a positive number")
        if K is None or K <= 0:
            raise ValueError("K must be a positive number")
        if T is None or T <= 0:
            raise ValueError("T must be a positive number")
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be a positive number")

    def CRR_Option(self, S0=None, r=None, sigma=None, T=None, M=None, K=None,
                   option_type='call', option_style='European'):
        S0 = self.S0 if S0 is None else S0
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        T = self.T if T is None else T
        M = self.M if M is None else M
        K = self.K if K is None else K
        self._validate(S0, K, T, sigma, option_type)
        if option_style not in ('European', 'American'):
            raise ValueError("option_style must be 'European' or 'American'")

        delta_t = T / M
        beta = (np.exp(-r * delta_t) + np.exp((r + sigma ** 2) * delta_t)) / 2
        u = beta + np.sqrt(beta ** 2 - 1)
        d = 1 / u
        q = (np.exp(r * delta_t) - d) / (u - d)

        S = np.zeros((M + 1, M + 1))
        for i in range(M + 1):
            for j in range(i + 1):
                S[j, i] = S0 * (u ** j) * (d ** (i - j))

        V = np.zeros((M + 1, M + 1))
        if option_type == 'call':
            V[:, M] = np.maximum(S[:, M] - K, 0.0)
        else:
            V[:, M] = np.maximum(K - S[:, M], 0.0)

        discount = np.exp(-r * delta_t)
        for i in range(M - 1, -1, -1):
            for j in range(i + 1):
                hold = discount * (q * V[j + 1, i + 1] + (1 - q) * V[j, i + 1])
                if option_style == 'European':
                    V[j, i] = hold
                else:
                    ex = max(S[j, i] - K, 0.0) if option_type == 'call' else max(K - S[j, i], 0.0)
                    V[j, i] = max(ex, hold)
        return V[0, 0]

    def BlackScholes_Option(self, t=0, St=None, K=None, T=None, r=None, sigma=None,
                            option='call', option_style='European'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        self._validate(St, K, T - t, sigma, option)

        if option_style == 'American':
            return self.CRR_Option(S0=St, r=r, sigma=sigma, T=T - t,
                                   M=max(self.M, 500), K=K,
                                   option_type=option, option_style='American')

        tau = T - t
        d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        if option == 'call':
            return St * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        return K * np.exp(-r * tau) * norm.cdf(-d2) - St * norm.cdf(-d1)

    def Greeks(self, t=0, St=None, K=None, T=None, r=None, sigma=None, option='call'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        self._validate(St, K, T - t, sigma, option)

        tau = T - t
        d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        delta = norm.cdf(d1) if option == 'call' else norm.cdf(d1) - 1.0
        gamma = norm.pdf(d1) / (St * sigma * np.sqrt(tau))
        vega = St * norm.pdf(d1) * np.sqrt(tau)
        if option == 'call':
            theta = -(St * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * norm.cdf(d2)
            rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
        else:
            theta = -(St * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * norm.cdf(-d2)
            rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def MonteCarlo_Option(self, t=0, St=None, K=None, T=None, r=None, sigma=None,
                          M=None, option='call', seed=42):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        M = 10000 if M is None else M
        self._validate(St, K, T - t, sigma, option)

        rng = np.random.default_rng(seed)
        Z = rng.normal(0, 1, M)
        ST = St * np.exp((r - 0.5 * sigma ** 2) * (T - t) + sigma * np.sqrt(T - t) * Z)
        payoff = np.maximum(ST - K, 0.0) if option == 'call' else np.maximum(K - ST, 0.0)
        price = np.exp(-r * (T - t)) * np.mean(payoff)
        stderr = np.exp(-r * (T - t)) * np.std(payoff, ddof=1) / np.sqrt(M)
        ci = (price - 1.96 * stderr, price + 1.96 * stderr)
        return price, stderr, ci

    def BS_Price_Int(self, t=0, St=None, K=None, T=None, r=None, sigma=None, option='call'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        self._validate(St, K, T - t, sigma, option)

        tau = T - t

        def integrand(x):
            ST = St * np.exp((r - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * x)
            payoff = max(ST - K, 0.0) if option == 'call' else max(K - ST, 0.0)
            return payoff * np.exp(-0.5 * x ** 2)

        val, _ = quad(integrand, -np.inf, np.inf)
        return np.exp(-r * tau) / np.sqrt(2 * np.pi) * val

    def laplace_BS(self, t=0, St=None, K=None, T=None, r=None, sigma=None, option='call'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        self._validate(St, K, T - t, sigma, option)

        tau = T - t

        def integrand(u):
            phi = np.exp(1j * u * (np.log(St) + (r - 0.5 * sigma ** 2) * tau)
                         - 0.5 * sigma ** 2 * u ** 2 * tau)
            return np.real(np.exp(-1j * u * np.log(K)) * phi) / (u ** 2 + 0.25)

        integral, _ = quad(integrand, 0, 200, limit=200)
        call_price = St - K * np.exp(-r * tau) / np.pi * integral
        if option == 'call':
            return call_price
        return call_price - St + K * np.exp(-r * tau)

    def AmPerpPut_ODE(self, S0=None, K=None, r=None, sigma=None):
        S0 = self.S0 if S0 is None else S0
        K = self.K if K is None else K
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        if S0 is None or S0 <= 0 or K is None or K <= 0:
            raise ValueError("S0 and K must be positive")
        if r is None or r <= 0 or sigma is None or sigma <= 0:
            raise ValueError("r and sigma must be positive")

        gamma = 2.0 * r / sigma ** 2
        S_star = K * gamma / (gamma + 1.0)
        if S0 <= S_star:
            return K - S0
        return (K - S_star) * (S0 / S_star) ** (-gamma)

    def BS_EuOption_FiDi_Explicit(self, S0=None, K=None, r=None, sigma=None, T=None,
                                   a=-2.0, b=2.0, m=200, nu_max=2000, option='call'):
        S0 = self.S0 if S0 is None else S0
        K = self.K if K is None else K
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        T = self.T if T is None else T
        self._validate(S0, K, T, sigma, option)

        q = 2.0 * r / sigma ** 2
        dx = (b - a) / m
        dt = T * sigma ** 2 / (2.0 * nu_max)
        lam = dt / dx ** 2
        if lam > 0.5:
            raise ValueError(f"Explicit scheme unstable: lambda={lam:.4f} > 0.5. Increase nu_max or decrease m")

        x = np.linspace(a, b, m + 1)
        t_grid = np.linspace(0, nu_max * dt, nu_max + 1)

        w = np.zeros((nu_max + 1, m + 1))
        if option == 'call':
            w[0] = np.maximum(np.exp(0.5 * x * (q + 1)) - np.exp(0.5 * x * (q - 1)), 0.0)
        else:
            w[0] = np.maximum(np.exp(0.5 * x * (q - 1)) - np.exp(0.5 * x * (q + 1)), 0.0)

        A = (np.diag(np.repeat(lam, m - 2), -1)
             + np.diag(np.repeat(1 - 2 * lam, m - 1))
             + np.diag(np.repeat(lam, m - 2), 1))

        d = np.zeros(m - 1)
        for nu in range(nu_max):
            tau_orig = T - (2.0 / sigma ** 2) * t_grid[nu]
            S_right = K * np.exp(x[-1])
            if option == 'call':
                v_right = S_right - K * np.exp(-r * (T - tau_orig))
            else:
                v_right = 0.0
            w_right = (v_right / K) * np.exp(
                0.5 * (q - 1) * x[-1]
                + 0.5 * sigma ** 2 * (T - tau_orig) * ((q - 1) ** 2 / 4 + q)
            )
            d[-1] = lam * w_right
            w[nu + 1, 1:-1] = A @ w[nu, 1:-1] + d

        S_grid = K * np.exp(x[1:-1])
        t_orig = T - (2.0 / sigma ** 2) * t_grid
        V = K * w * np.exp(-x[1:-1] / 2 * (q - 1)
                           - sigma ** 2 / 2 * (T - t_orig[:, None])
                           * ((q - 1) ** 2 / 4 + q))
        idx = np.argmin(np.abs(S_grid - S0))
        return V[-1, idx]

    def laplace_heston(self, t=0, St=None, K=None, T=None, r=None,
                       gam0=None, kappa=None, lamb=None, sig_tilde=None, rho=None,
                       R=1.5, upper=100, option='call'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r
        gam0 = self.gam0 if gam0 is None else gam0
        kappa = self.kappa if kappa is None else kappa
        lamb = self.lamb if lamb is None else lamb
        sig_tilde = self.sig_tilde if sig_tilde is None else sig_tilde
        rho = self.rho if rho is None else rho
        self._validate(St, K, T - t, sig_tilde, option)

        tau = T - t

        def heston_char(u):
            B = rho * sig_tilde * u * 1j - lamb
            D = np.sqrt(B ** 2 + sig_tilde ** 2 * u * (u + 1j))
            lnG = np.log(B * (np.exp(-D * tau) - 1) / (2 * D) + (np.exp(-D * tau) + 1) / 2)
            psi_0 = u * 1j * r * tau - kappa / sig_tilde ** 2 * ((B + D) * tau + 2 * lnG)
            psi_1 = u * 1j
            psi_2 = u * (u + 1j) * (np.exp(-D * tau) - 1) / (
                B * (np.exp(-D * tau) - 1) + D * (np.exp(-D * tau) + 1)
            )
            return np.exp(psi_0 + psi_1 * np.log(St) + psi_2 * gam0)

        def integrand(u):
            z = R + 1j * u
            ft = K ** (1 - z) / (z * (z - 1))
            return np.real(ft * heston_char(u - 1j * R))

        integral, _ = quad(integrand, 0, upper, limit=200)
        call_price = np.exp(-r * tau) / np.pi * integral
        if option == 'call':
            return call_price
        return call_price - St + K * np.exp(-r * tau)

    def implied_volatility(self, market_price, t=0, St=None, K=None, T=None,
                           r=None, option='call'):
        St = self.S0 if St is None else St
        K = self.K if K is None else K
        T = self.T if T is None else T
        r = self.r if r is None else r

        def objective(sigma):
            return self.BlackScholes_Option(t=t, St=St, K=K, T=T, r=r,
                                            sigma=sigma, option=option) - market_price

        try:
            return brentq(objective, 1e-6, 5.0, xtol=1e-10, maxiter=200)
        except ValueError:
            return np.nan

    def fetch_stock_data(self, ticker, period='1y', interval='1d'):
        if yf is None:
            raise ImportError("yfinance is required. Install via pip install yfinance")
        return yf.download(ticker, period=period, interval=interval, progress=False)

    def fetch_option_chain(self, ticker, expiration=None):
        if yf is None:
            raise ImportError("yfinance is required. Install via pip install yfinance")
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        expiration = expirations[0] if expiration is None else expiration
        chain = tk.option_chain(expiration)
        return chain.calls, chain.puts, expiration

    def check_arbitrage(self, calls, puts, S0, r, T, tol=0.5):
        calls = calls.sort_values('strike').reset_index(drop=True)
        puts = puts.sort_values('strike').reset_index(drop=True)
        violations = []

        for i in range(len(calls)):
            K_i = calls.loc[i, 'strike']
            c_i = calls.loc[i, 'lastPrice']
            lower_c = max(S0 - K_i * np.exp(-r * T), 0.0)
            if c_i < lower_c - tol:
                violations.append(('call_lower_bound', K_i, c_i, lower_c))

            if i < len(puts):
                K_p = puts.loc[i, 'strike']
                p_i = puts.loc[i, 'lastPrice']
                lower_p = max(K_p * np.exp(-r * T) - S0, 0.0)
                if p_i < lower_p - tol:
                    violations.append(('put_lower_bound', K_p, p_i, lower_p))

                if abs(K_i - K_p) < 1e-6:
                    parity = S0 - K_i * np.exp(-r * T)
                    diff = (c_i - p_i) - parity
                    if abs(diff) > tol:
                        violations.append(('put_call_parity', K_i, c_i - p_i, parity))

        return violations
