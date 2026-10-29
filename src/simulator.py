"""Lump-sum / SIP engine with rebalancing, trading costs and fund fees."""
import pandas as pd

MARKETS = ["India", "USA", "Korea"]


def _rebalance_now(date, i, rule):
    if i == 0 or rule == "Never":
        return False
    if rule == "Yearly":
        return date.month == 12
    return date.month in (3, 6, 9, 12)  # Quarterly


def simulate(prices, weights, amount, mode="lump", rebalance="Yearly",
             cost=0.002, fees=None):
    """Month-by-month portfolio value in INR.

    prices: monthly INR prices for the chosen dates.
    weights: {"India": 0.4, "USA": 0.4, "Korea": 0.2}, summing to 1.
    amount: lump-sum amount, or the monthly SIP amount.
    fees: yearly fee per market as a decimal, e.g. {"USA": 0.005, ...}.
    """
    w = pd.Series(weights)[MARKETS]
    fee_rates = pd.Series(fees or {m: 0.0 for m in MARKETS})[MARKETS]
    rets = prices[MARKETS].pct_change().fillna(0)
    hold = pd.Series(0.0, index=MARKETS)
    invested, rows = 0.0, []

    for i, date in enumerate(rets.index):
        hold = hold * (1 + rets.loc[date]) * (1 - fee_rates / 12)  # grow, pay fee
        if (mode == "lump" and i == 0) or mode == "sip":
            hold += amount * w                                     # new money
            invested += amount
        if _rebalance_now(date, i, rebalance):
            traded = (hold.sum() * w - hold).abs().sum()
            hold = (hold.sum() - traded * cost) * w                # reset weights
        rows.append({"date": date, "invested": invested,
                     "value": hold.sum(), **hold.to_dict()})
    return pd.DataFrame(rows).set_index("date")
