"""Automatic split optimiser: pick weights from historic INR returns.

Objectives: maximum Sharpe ratio, minimum volatility, or risk parity. Weights
are bounded so three assets cannot collapse into one. A walk-forward backtest
re-optimises each January using only data before that month, so the result is
an honest out-of-sample test rather than a fit to the same history.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.metrics import ann_vol, cagr, max_drawdown, sharpe
from src.simulator import MARKETS

OBJECTIVES = ["Max Sharpe", "Min volatility", "Risk parity"]


def history_stats(data, years):
    """Annualised mean return vector and covariance from the last `years`."""
    rets = data[MARKETS].pct_change().dropna().tail(12 * years)
    mu = ((1 + rets.mean()) ** 12 - 1).values
    cov = rets.cov().values * 12
    return mu, cov


def portfolio_stats(w, mu, cov, rf):
    """(annual return, annual volatility, Sharpe) for weights `w`."""
    ret, vol = float(w @ mu), float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol


def _objective(name, mu, cov, rf):
    if name == "Max Sharpe":
        return lambda w: -portfolio_stats(w, mu, cov, rf)[2]
    if name == "Min volatility":
        return lambda w: w @ cov @ w
    if name == "Risk parity":
        def spread(w):  # penalise unequal risk contributions
            contrib = w * (cov @ w) / (w @ cov @ w)
            return float(((contrib - 1 / len(w)) ** 2).sum())
        return spread
    raise ValueError(f"Unknown objective: {name}")


def bounds_for(min_w=0.0, max_w=0.7, min_india=0.2):
    return [(max(min_w, min_india), max_w), (min_w, max_w), (min_w, max_w)]


def optimise(data, objective="Max Sharpe", years=5, rf=0.065, bounds=None):
    """Best weights (Series, sums to 1) and their in-sample stats.

    Returns (weights, (return, volatility, Sharpe)).
    """
    bounds = bounds or bounds_for()
    mu, cov = history_stats(data, years)
    fun = _objective(objective, mu, cov, rf)
    best = None
    for x0 in (np.ones(3) / 3, np.array([.6, .2, .2]),
               np.array([.2, .6, .2]), np.array([.2, .2, .6])):
        res = minimize(fun, x0=np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]),
                       bounds=bounds, method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Optimiser did not converge for these constraints")
    w = pd.Series(best.x, index=MARKETS).clip(lower=0)
    w = w / w.sum()
    return w, portfolio_stats(w.values, mu, cov, rf)


def efficient_frontier(data, years=5, rf=0.065, bounds=None, n=4000, seed=0):
    """Random feasible portfolios: DataFrame of weights, return, vol, Sharpe."""
    bounds = bounds or bounds_for()
    mu, cov = history_stats(data, years)
    raw = np.random.default_rng(seed).dirichlet(np.ones(3), size=n)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    raw = raw[((raw >= lo) & (raw <= hi)).all(axis=1)]
    rows = [(*w, *portfolio_stats(w, mu, cov, rf)) for w in raw]
    return pd.DataFrame(rows, columns=[*MARKETS, "Return", "Volatility", "Sharpe"])


def round_to_percent(weights):
    """Whole-number percents that sum to exactly 100 (India absorbs rounding)."""
    pct = (weights * 100).round().astype(int)
    pct["India"] += 100 - pct.sum()
    return pct


def walk_forward(data, start, end, objective="Max Sharpe", years=5, rf=0.065,
                 amount=100000, cost=0.002, bounds=None):
    """Re-optimise each January from data strictly before it; track the value.

    Until there are `years` of history, equal weights are used. Turnover cost
    is charged on weight changes. Gross of fund fees. Returns
    (growth DataFrame, weights-by-year DataFrame).
    """
    window = data.loc[pd.Timestamp(start):pd.Timestamp(end)]
    rets = window[MARKETS].pct_change().fillna(0)
    value, w = float(amount), pd.Series(1 / 3, index=MARKETS)
    values, weight_log = [], {}
    for i, date in enumerate(rets.index):
        if i == 0 or date.month == 1:
            past = data.loc[:date].iloc[:-1]
            if len(past) >= 12 * years + 1:
                new_w, _ = optimise(past, objective, years, rf, bounds)
            else:
                new_w = pd.Series(1 / 3, index=MARKETS)
            value *= 1 - cost * (new_w - w).abs().sum() if i else 1
            w = new_w
            weight_log[date.year] = w
        value *= 1 + float(rets.loc[date] @ w)
        values.append(value)
    equal = amount * (1 + rets.mean(axis=1)).cumprod()
    nifty = amount * (1 + rets["India"]).cumprod()
    growth = pd.DataFrame({"Auto-optimised": values, "Equal weight": equal.values,
                           "100% Nifty": nifty.values}, index=rets.index)
    return growth, pd.DataFrame(weight_log).T


def summarise(growth, rf=0.065):
    """CAGR, volatility, Sharpe and max drawdown for each growth column."""
    rows = {}
    for col in growth:
        r = growth[col].pct_change().dropna()
        rows[col] = {"CAGR": cagr(growth[col]), "Volatility": ann_vol(r),
                     "Sharpe": sharpe(r, rf), "Max drawdown": max_drawdown(growth[col]),
                     "Final value": growth[col].iloc[-1]}
    return pd.DataFrame(rows).T
