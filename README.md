# Global Portfolio Simulator: India vs USA vs South Korea

An interactive Python/Streamlit backtester. Enter an amount, dates and a split
between Nifty 50, S&P 500 and KOSPI, and see what the money would have become in
rupees. It separates market return from currency return and suggests a
lower-risk split. It is a historical backtest, not a predictor.

**Live app: https://global-portfolio-sim-lslccdayqgcwhhj42vhmnr.streamlit.app/**

## Questions the app answers
1. What would ₹X (lump sum or monthly SIP) have grown to in INR?
2. Which market gave the best return per unit of risk?
3. How much of the USA/Korea return came from the rupee moving?
4. Does mixing the markets lower risk versus Nifty alone?
5. What lower-risk split does history suggest (and did it work out of sample)?

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Run tests with `python -m pytest`.

## Structure
```
app.py            Streamlit page (sidebar inputs, charts, downloads)
src/data.py       yfinance download, INR conversion, CSV cache
src/metrics.py    CAGR, volatility, Sharpe, drawdown, best/worst year, XIRR
src/simulator.py  lump sum / SIP / rebalancing / costs engine
src/analysis.py   currency split, minimum-variance split, honesty check
src/optimizer.py  auto-optimiser, efficient frontier, walk-forward test
src/export.py     Excel export
tests/            checks from the project guide
data/             cached monthly prices (auto-created)
```

## Data
yfinance: ^NSEI, ^GSPC, ^KS11, USDINR=X, KRW=X. Month-end closes on common dates.
USD and KRW indices are multiplied by the INR rate (KRWINR = USDINR / USDKRW).
Common start date is **Sep 2007** (Nifty history on yfinance starts then).
Delete `data/prices_inr_monthly.csv` or use the sidebar button to refresh.

## Method
- **Simulator:** monthly growth, monthly fund fee (yearly fee / 12), new money
  each month for SIP, optional rebalancing with a trading cost on money moved.
- **Currency split:** (1 + r_INR) = (1 + r_local)(1 + r_FX), shown as CAGR parts.
- **Suggestion:** minimum-variance weights (SciPy) with max 70% per market and
  min 20% India, from the last N years of monthly INR returns.

- **Auto-optimiser:** pick a goal (max Sharpe, min volatility or risk parity)
  and bounds; SciPy SLSQP finds the split from the last N years of INR returns.
  A walk-forward test re-optimises each January using only earlier data and
  compares against equal weight and Nifty, so the result is out of sample.

## Limitations
Price indices exclude dividends (~1-2%/yr). Past returns do not predict future
ones. Taxes and LRS/TCS are not modelled. The last month may be a partial month.

## Disclaimer
Educational project, not financial advice. Consult a SEBI-registered advisor
before investing real money.
