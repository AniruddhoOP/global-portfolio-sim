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

## Key findings
All in INR, default costs (0.2% per trade, fund fees 0.2/0.5/0.6%), price
indices. Sharpe and currency figures use January 2010 to October 2026.

- **Risk-adjusted return:** USA Sharpe 0.89 (best), Korea 0.36, India 0.28
  (worst) at a 6.5% risk-free rate.
- **Currency:** the rupee's fall added 4.43 points a year to the S&P 500's
  17.47% INR return (about 25%), and 3.53 points to KOSPI's 12.03% (about 29%),
  so the won helped too.
- **2008:** a 34/33/33 mix fell 39.7% at its worst versus 47.3% for Nifty alone.
- **March 2020:** the mix fell 19.9% versus 28.1% for Nifty alone.
- **Lump sum vs SIP:** the same ₹20.2 lakh ended at ₹1.67 crore as a lump sum
  and ₹73.2 lakh as a SIP, but that is mostly because the lump sum is invested
  throughout. The rates are similar: 13.45% a year (lump sum) vs 13.95% XIRR (SIP).
- **Auto-optimiser (walk-forward, 5-year lookback, from 2010):** min volatility
  had 12.5% volatility and Sharpe 0.70 versus 14.5% and 0.59 for equal weight,
  and 16.4% and 0.28 for Nifty alone. Max Sharpe cut volatility too but ended
  with slightly less money than equal weight (₹8.68 vs ₹8.76 lakh). One period
  only, so treat it as suggestive.

## Limitations
Price indices exclude dividends (~1-2%/yr). Past returns do not predict future
ones. Taxes and LRS/TCS are not modelled. The last month may be a partial month.

## Disclaimer
Educational project, not financial advice. Consult a SEBI-registered advisor
before investing real money.
