"""Download index and FX data from yfinance and convert everything to INR."""
import os
import warnings

import pandas as pd
import yfinance as yf

CACHE = os.path.join("data", "prices_inr_monthly.csv")
TICKERS = ["^NSEI", "^GSPC", "^KS11", "USDINR=X", "KRW=X"]


def _download(start):
    """Month-end closes for all tickers, common dates only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # yfinance is noisy on new pandas
        daily = yf.download(TICKERS, start=start, auto_adjust=True,
                            progress=False)["Close"]
    if daily.empty:
        raise RuntimeError("yfinance returned no data")
    daily = daily.sort_index().ffill(limit=5)  # bridge different holidays
    return daily.resample("ME").last().dropna()


def _to_inr(m):
    """Build local-currency, FX and INR columns from month-end closes."""
    out = pd.DataFrame({
        "India_local": m["^NSEI"],
        "USA_local": m["^GSPC"],
        "Korea_local": m["^KS11"],
        "USDINR": m["USDINR=X"],                    # INR per USD
        "KRWINR": m["USDINR=X"] / m["KRW=X"],       # INR per KRW (cross rate)
    })
    out["India"] = out["India_local"]
    out["USA"] = out["USA_local"] * out["USDINR"]
    out["Korea"] = out["Korea_local"] * out["KRWINR"]
    return out


def load_data(start="2005-01-01", refresh=False):
    """Monthly prices (local, FX, INR). Uses the CSV cache unless refreshing.

    If a refresh fails (offline / yfinance down) the cached file is used.
    """
    cached = os.path.exists(CACHE)
    if cached and not refresh:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    try:
        out = _to_inr(_download(start))
    except Exception:
        if cached:
            return pd.read_csv(CACHE, index_col=0, parse_dates=True)
        raise
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out.to_csv(CACHE)
    return out
