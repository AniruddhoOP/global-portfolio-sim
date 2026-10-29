"""Currency-vs-index split, minimum-variance suggestion and honesty check."""
import pandas as pd

from src.metrics import cagr
from src.optimizer import bounds_for, optimise
from src.simulator import MARKETS, simulate

FX_COLUMN = {"USA": "USDINR", "Korea": "KRWINR"}


def currency_split(data):
    """Annualised (CAGR) contribution of index vs currency for USA and Korea."""
    out = {}
    for m, fx in FX_COLUMN.items():
        local, cur, inr = cagr(data[f"{m}_local"]), cagr(data[fx]), cagr(data[m])
        out[m] = {"Index return": local, "Currency effect": cur,
                  "Cross term": inr - local - cur, "Total INR return": inr}
    return pd.DataFrame(out).T


def suggest_split(data, years=5, max_w=0.7, min_india=0.2):
    """Lowest-risk split using the last `years` of monthly INR returns."""
    weights, _ = optimise(data, "Min volatility", years,
                          bounds=bounds_for(0.0, max_w, min_india))
    return weights.round(2)


def honesty_check(data, start, end, years, amount, rebalance, cost, fees):
    """Suggest a split from data BEFORE `start`, then simulate it forward.

    Returns (weights, final value of suggestion, final value of equal weight)
    or None when there is not enough history before `start`.
    """
    past = data.loc[:pd.Timestamp(start)].iloc[:-1]  # strictly before start
    if len(past) < 12 * years + 1:
        return None
    weights = suggest_split(past, years=years)
    weights = weights / weights.sum()  # rounding can leave 0.99 / 1.01
    window = data.loc[pd.Timestamp(start):pd.Timestamp(end)]
    mine = simulate(window, weights.to_dict(), amount, "lump", rebalance, cost, fees)
    equal = simulate(window, {m: 1 / 3 for m in MARKETS}, amount, "lump",
                     rebalance, cost, fees)
    return weights, mine["value"].iloc[-1], equal["value"].iloc[-1]
