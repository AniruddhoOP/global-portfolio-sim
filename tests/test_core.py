"""Checks from the project guide's testing checklist."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import currency_split, suggest_split
from src.data import load_data
from src.metrics import cagr, xirr
from src.optimizer import (OBJECTIVES, bounds_for, history_stats, optimise,
                           portfolio_stats, round_to_percent, walk_forward)
from src.simulator import MARKETS, simulate

NO_FEES = {m: 0.0 for m in MARKETS}


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_cagr_hand_check():
    idx = pd.to_datetime(["2020-01-01", "2025-01-01"])
    assert cagr(pd.Series([100, 200], index=idx)) == pytest.approx(0.1487, abs=2e-3)


def test_lump_sum_matches_nifty(data):
    prices = data.loc["2010-01-01":]
    sim = simulate(prices, {"India": 1, "USA": 0, "Korea": 0}, 100000, "lump",
                   "Never", 0, NO_FEES)
    expected = 100000 * prices["India"].iloc[-1] / prices["India"].iloc[0]
    assert sim["value"].iloc[-1] == pytest.approx(expected)


def test_sip_invested_total(data):
    prices = data.loc["2015-01-01":]
    sim = simulate(prices, {m: 1 / 3 for m in MARKETS}, 10000, "sip")
    assert sim["invested"].iloc[-1] == 10000 * len(prices)


def test_higher_costs_lower_value(data):
    prices = data.loc["2010-01-01":]
    w = {m: 1 / 3 for m in MARKETS}
    cheap = simulate(prices, w, 100000, "lump", "Quarterly", 0.0, NO_FEES)
    dear = simulate(prices, w, 100000, "lump", "Quarterly", 0.01,
                    {m: 0.01 for m in MARKETS})
    assert dear["value"].iloc[-1] < cheap["value"].iloc[-1]


def test_currency_split_adds_up(data):
    split = currency_split(data.loc["2010-01-01":])
    parts = split[["Index return", "Currency effect", "Cross term"]].sum(axis=1)
    assert np.allclose(parts, split["Total INR return"])


def test_suggested_weights_sum_to_one(data):
    w = suggest_split(data)
    assert w.sum() == pytest.approx(1, abs=0.02)
    assert w["India"] >= 0.2 - 1e-9


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_optimiser_respects_constraints(data, objective):
    bounds = bounds_for(0.0, 0.6, 0.25)
    w, _ = optimise(data, objective, 5, 0.065, bounds)
    assert w.sum() == pytest.approx(1)
    assert w["India"] >= 0.25 - 1e-6 and w.max() <= 0.6 + 1e-6


def test_max_sharpe_beats_equal_weight_in_sample(data):
    w, (_, _, sh) = optimise(data, "Max Sharpe", 5)
    mu, cov = history_stats(data, 5)
    assert sh >= portfolio_stats(np.ones(3) / 3, mu, cov, 0.065)[2] - 1e-9


def test_min_vol_is_lowest_volatility(data):
    _, (_, vol_min, _) = optimise(data, "Min volatility", 5)
    _, (_, vol_sharpe, _) = optimise(data, "Max Sharpe", 5)
    assert vol_min <= vol_sharpe + 1e-9


def test_percent_rounding_sums_to_100(data):
    w, _ = optimise(data, "Risk parity", 5)
    assert round_to_percent(w).sum() == 100


def test_walk_forward_has_no_lookahead(data):
    # Each January's weights must equal an optimisation on data strictly before it.
    _, log = walk_forward(data, "2012-01-01", "2020-12-31", years=3)
    jan = pd.Timestamp("2016-01-31")
    expected, _ = optimise(data.loc[:jan].iloc[:-1], "Max Sharpe", 3)
    assert np.allclose(log.loc[2016].values, expected.values, atol=1e-6)


def test_xirr_simple():
    flows = [(pd.Timestamp("2020-01-01"), -100), (pd.Timestamp("2021-01-01"), 110)]
    assert xirr(flows) == pytest.approx(0.10, abs=2e-3)
