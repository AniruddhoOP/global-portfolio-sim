"""Performance and risk metrics."""
import numpy as np
from scipy.optimize import brentq


def years_between(index):
    return (index[-1] - index[0]).days / 365.25


def cagr(values):
    """values = a price or portfolio-value series."""
    return (values.iloc[-1] / values.iloc[0]) ** (1 / years_between(values.index)) - 1


def ann_vol(rets):
    """rets = monthly returns."""
    return rets.std() * np.sqrt(12)


def sharpe(rets, rf=0.065):
    ann_ret = (1 + rets.mean()) ** 12 - 1
    return (ann_ret - rf) / ann_vol(rets)


def max_drawdown(values):
    return (values / values.cummax() - 1).min()


def calendar_year_returns(values):
    """Return per calendar year, from year-end to year-end values."""
    return values.groupby(values.index.year).last().pct_change().dropna()


def best_worst_year(values):
    """(best, worst) calendar-year return; (nan, nan) if under 2 full years."""
    yearly = calendar_year_returns(values)
    if yearly.empty:
        return np.nan, np.nan
    return yearly.max(), yearly.min()


def xirr(cashflows):
    """cashflows = list of (date, amount); money in is negative, final value positive."""
    t0 = cashflows[0][0]

    def npv(r):
        return sum(a / (1 + r) ** ((d - t0).days / 365.25) for d, a in cashflows)

    return brentq(npv, -0.99, 10)
