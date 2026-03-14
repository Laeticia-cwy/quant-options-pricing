# Options Pricing Portfolio

A 3-notebook quantitative finance project covering three approaches to pricing options on real market data.

## Structure

```
options_utils.py              ← shared library (all notebooks import from here)
1_black_scholes_pricing.ipynb ← closed-form pricing fundamentals
2_greeks_implied_vol.ipynb    ← risk sensitivity + implied volatility solver
3_monte_carlo_options.ipynb   ← simulation-based pricing + exotic options
```

## How They Connect

All three notebooks import from `options_utils.py` : one source of truth for pricing functions, Greeks, and data fetching. Each notebook uses live AAPL data fetched via `yfinance`.

| Notebook | Builds on previous by... |
|----------|--------------------------|
| 1 — Black-Scholes | Establishes benchmark prices and σ estimate |
| 2 — Greeks & IV | Takes NB1 prices and asks "how sensitive are they?" |
| 3 — Monte Carlo | Verifies NB1 prices via simulation, then goes beyond BS to exotic options |

## Topics Covered

**Notebook 1 — Black-Scholes Pricing**
- European call & put pricing (closed form)
- Put-Call Parity verification
- Price surface across strike × expiry grid

**Notebook 2 — Greeks & Implied Volatility**
- Delta, Gamma, Vega, Theta
- Greeks dashboard (4-chart visualization)
- Implied volatility solver (Brent's method)
- Volatility smile construction

**Notebook 3 — Monte Carlo**
- Geometric Brownian Motion path simulation
- European option pricing via MC, benchmarked against BS
- Convergence analysis (paths vs. accuracy)
- Exotic options: Asian (average price), Down-and-Out Barrier, Lookback

## Setup

```bash
pip install numpy scipy matplotlib yfinance
jupyter notebook
```

Run notebooks in order (1 → 2 → 3) for the full narrative, or run any notebook standalone.
