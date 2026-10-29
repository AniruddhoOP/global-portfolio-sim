"""Checks from the project guide's testing checklist."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import currency_split, suggest_split
from src.data import load_data
from src.metrics import cagr, xirr
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


def test_xirr_simple():
    flows = [(pd.Timestamp("2020-01-01"), -100), (pd.Timestamp("2021-01-01"), 110)]
    assert xirr(flows) == pytest.approx(0.10, abs=2e-3)
