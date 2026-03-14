"""
options_utils.py
================
Shared library for the options pricing portfolio.
Imported by all 3 notebooks — defines pricing, Greeks, and data fetching once.
"""

import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
import yfinance as yf


# ── DATA ───────────────────────────────────────────────────────────────────

def fetch_market_data(ticker="AAPL"):
    """
    Fetch current price and annualized historical volatility.
    Returns dict with S, sigma, ticker, history.
    """
    data = yf.Ticker(ticker)
    history = data.history(period="1y")['Close']
    S = history.iloc[-1]
    sigma = np.std(history.pct_change().dropna()) * np.sqrt(252)
    return {"ticker": ticker, "S": S, "sigma": sigma, "history": history}


# ── BLACK-SCHOLES ──────────────────────────────────────────────────────────

def d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call(S, K, T, r, sigma):
    """European call price (Black-Scholes closed form)."""
    d1, d2 = d1d2(S, K, T, r, sigma)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    """European put price (Black-Scholes closed form)."""
    d1, d2 = d1d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)


# ── GREEKS ─────────────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option="call"):
    d1, _ = d1d2(S, K, T, r, sigma)
    return si.norm.cdf(d1) if option == "call" else si.norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1, _ = d1d2(S, K, T, r, sigma)
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """Vega per 1% change in vol."""
    d1, _ = d1d2(S, K, T, r, sigma)
    return S * si.norm.pdf(d1) * np.sqrt(T) * 0.01

def theta(S, K, T, r, sigma, option="call"):
    """Daily theta (time decay per calendar day)."""
    d1, d2 = d1d2(S, K, T, r, sigma)
    term1 = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option == "call":
        return (term1 - r * K * np.exp(-r * T) * si.norm.cdf(d2))  / 365
    else:
        return (term1 + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365


# ── IMPLIED VOLATILITY ─────────────────────────────────────────────────────

def implied_vol(market_price, S, K, T, r, option="call", tol=1e-6):
    """
    Recover implied volatility from a market price using Brent's method.
    Returns NaN if no solution found.
    """
    pricer = bs_call if option == "call" else bs_put
    intrinsic = max(S - K, 0) if option == "call" else max(K - S, 0)
    if market_price <= intrinsic:
        return np.nan
    try:
        return brentq(lambda s: pricer(S, K, T, r, s) - market_price, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return np.nan


# ── MONTE CARLO ────────────────────────────────────────────────────────────

def simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """
    Simulate GBM paths. Returns array shape (n_steps+1, n_paths).
    Used by notebook 3 — and for convergence checks in notebook 1.
    """
    np.random.seed(seed)
    dt = T / n_steps
    Z  = np.random.standard_normal((n_steps, n_paths))
    paths = np.ones((n_steps + 1, n_paths)) * S0
    for t in range(1, n_steps + 1):
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
    return paths

def mc_price(S0, K, T, r, sigma, option="call", n_paths=100_000, seed=42):
    """
    Price a European option via Monte Carlo.
    Returns (price, standard_error).
    """
    np.random.seed(seed)
    Z   = np.random.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0) if option == "call" else np.maximum(K - S_T, 0)
    price  = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, stderr
