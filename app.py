"""Streamlit front end: sidebar inputs -> simulation -> charts and tables."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import currency_split, honesty_check, suggest_split
from src.data import load_data
from src.export import to_excel
from src.metrics import (ann_vol, best_worst_year, cagr, max_drawdown, sharpe,
                         xirr)
from src.optimizer import (OBJECTIVES, bounds_for, efficient_frontier,
                           history_stats, optimise, portfolio_stats,
                           round_to_percent, summarise, walk_forward)
from src.simulator import MARKETS, simulate

COLORS = {"India": "#E8833A", "USA": "#2F6FDE", "Korea": "#2BA38B",
          "Your mix": "#7A3FD1", "100% Nifty": "#E8833A",
          "Equal 33/33/33": "#8A8F98"}

st.set_page_config(page_title="Global Portfolio Simulator", page_icon="🌏",
                   layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  div[data-testid="stMetric"] {
      background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
      border-radius: 12px; padding: 14px 18px;}
  h1 {margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data(refresh=False):
    return load_data(refresh=refresh)


# ---------------------------------------------------------------- sidebar --
def read_inputs(first, last):
    """Render every sidebar widget and return the chosen settings."""
    sb = st.sidebar
    sb.header("Your investment")
    mode = sb.radio("Investment mode", ["Lump sum", "Monthly SIP"], horizontal=True)
    amount = sb.number_input("Amount (₹)", min_value=500, step=500,
                             value=100000 if mode == "Lump sum" else 10000)
    start = sb.date_input("Start", max(first, date(2010, 1, 1)), first, last)
    end = sb.date_input("End", last, first, last)

    sb.subheader("Allocation (%)")
    alloc = {m: sb.slider(m, 0, 100, d, key=f"alloc_{m}")
             for m, d in [("India", 34), ("USA", 33), ("Korea", 33)]}

    sb.subheader("Costs and rebalancing")
    rebalance = sb.selectbox("Rebalance", ["Never", "Yearly", "Quarterly"], index=1)
    cost = sb.number_input("Cost per trade (%)", 0.0, 2.0, 0.2) / 100
    fees = {m: sb.number_input(f"{m} fund fee (%/yr)", 0.0, 3.0, f) / 100
            for m, f in [("India", 0.2), ("USA", 0.5), ("Korea", 0.6)]}

    sb.subheader("Risk")
    rf = sb.number_input("Risk-free rate (%)", 0.0, 15.0, 6.5) / 100
    lookback = sb.slider("Years of history for suggestion", 3, 15, 5)

    if sb.button("Refresh market data"):
        st.cache_data.clear()
        get_data(refresh=True)
        st.rerun()
    return dict(mode="lump" if mode == "Lump sum" else "sip", amount=amount,
                start=start, end=end, alloc=alloc, rebalance=rebalance,
                cost=cost, fees=fees, rf=rf, lookback=lookback)


def validate(cfg):
    """Stop the script with a friendly message when inputs are unusable."""
    total = sum(cfg["alloc"].values())
    if total != 100:
        st.error(f"Allocation adds up to {total}%. Make it 100%.")
        st.stop()
    if cfg["start"] >= cfg["end"]:
        st.warning("Start date must be before the end date.")
        st.stop()


# ----------------------------------------------------------- calculations --
def run_simulations(prices, cfg):
    w = {k: v / 100 for k, v in cfg["alloc"].items()}
    args = (cfg["amount"], cfg["mode"])
    mine = simulate(prices, w, *args, cfg["rebalance"], cfg["cost"], cfg["fees"])
    nifty = simulate(prices, {"India": 1, "USA": 0, "Korea": 0}, *args,
                     "Never", cfg["cost"], cfg["fees"])
    equal = simulate(prices, {m: 1 / 3 for m in MARKETS}, *args,
                     cfg["rebalance"], cfg["cost"], cfg["fees"])
    return mine, nifty, equal


def headline_return(mine, cfg):
    """CAGR for a lump sum, XIRR for a SIP."""
    if cfg["mode"] == "lump":
        return "CAGR", cagr(mine["value"])
    flows = [(d, -cfg["amount"]) for d in mine.index] + \
            [(mine.index[-1], mine["value"].iloc[-1])]
    return "XIRR", xirr(flows)


def market_table(prices, rf):
    rets = prices[MARKETS].pct_change().dropna()
    rows = {}
    for mk in MARKETS:
        best, worst = best_worst_year(prices[mk])
        rows[mk] = {"CAGR": cagr(prices[mk]), "Volatility": ann_vol(rets[mk]),
                    "Sharpe": sharpe(rets[mk], rf),
                    "Max drawdown": max_drawdown(prices[mk]),
                    "Best year": best, "Worst year": worst}
    return pd.DataFrame(rows).T, rets


# ------------------------------------------------------------- renderers --
def render_headline(mine, label, ret):
    final, invested = mine["value"].iloc[-1], mine["invested"].iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invested", f"₹{invested:,.0f}")
    c2.metric("Final value", f"₹{final:,.0f}")
    c3.metric("Profit", f"₹{final - invested:,.0f}",
              f"{(final / invested - 1):.1%}")
    c4.metric(label, f"{ret:.2%}")


def render_growth(mine, nifty, equal):
    growth = pd.DataFrame({"Your mix": mine["value"], "100% Nifty": nifty["value"],
                           "Equal 33/33/33": equal["value"]})
    fig = px.line(growth, color_discrete_map=COLORS,
                  labels={"value": "₹", "date": "", "variable": ""})
    # Dotted benchmark keeps "Your mix" visible when the two nearly coincide.
    fig.update_traces(line_dash="dot", selector=dict(name="Equal 33/33/33"))
    fig.update_traces(line_width=3, selector=dict(name="Your mix"))
    fig.update_layout(hovermode="x unified", legend_orientation="h",
                      legend_y=1.1, margin=dict(t=30, b=0))
    st.plotly_chart(fig, width="stretch")


def render_markets(table, prices):
    st.dataframe(
        table.style.format("{:.2%}", subset=["CAGR", "Volatility", "Max drawdown",
                                             "Best year", "Worst year"])
                   .format("{:.2f}", subset=["Sharpe"]),
        width="stretch")
    rebased = prices[MARKETS] / prices[MARKETS].iloc[0] * 100
    fig = px.line(rebased, color_discrete_map=COLORS,
                  labels={"value": "Index (start = 100, in ₹)", "date": "",
                          "variable": ""})
    fig.update_layout(hovermode="x unified", legend_orientation="h",
                      legend_y=1.1, margin=dict(t=30, b=0))
    st.plotly_chart(fig, width="stretch")


def render_currency(prices):
    split = currency_split(prices)
    st.caption("Annualised. A positive currency effect means the rupee fell "
               "against that currency, which added to your INR return.")
    long = split[["Index return", "Currency effect", "Cross term"]] \
        .reset_index(names="Market").melt("Market", var_name="Source",
                                          value_name="Contribution")
    fig = px.bar(long, x="Market", y="Contribution", color="Source",
                 barmode="relative", text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=10, b=0))
    left, right = st.columns([1, 1])
    left.plotly_chart(fig, width="stretch")
    right.dataframe(split.style.format("{:.2%}"), width="stretch")


def render_correlation(rets):
    fig = px.imshow(rets.corr(), text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r")
    fig.update_layout(margin=dict(t=10, b=0))
    st.plotly_chart(fig, width="stretch")


def render_suggestion(data, cfg):
    sug = suggest_split(data.loc[:pd.Timestamp(cfg["end"])], years=cfg["lookback"])
    cols = st.columns(3)
    for col, mk in zip(cols, MARKETS):
        col.metric(mk, f"{sug[mk]:.0%}")
    st.caption("Based on past volatility and correlation only. Not a prediction "
               "or advice.")
    with st.expander("Honesty check: would this suggestion have beaten equal weight?"):
        result = honesty_check(data, cfg["start"], cfg["end"], cfg["lookback"],
                               cfg["amount"], cfg["rebalance"], cfg["cost"],
                               cfg["fees"])
        if result is None:
            st.info("Not enough data before your start date for this lookback. "
                    "Pick a later start date or fewer lookback years.")
            return
        w, mine_val, equal_val = result
        st.write("Split computed using only data before "
                 f"{cfg['start']}: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))
        a, b = st.columns(2)
        a.metric("Suggested split (lump sum, final)", f"₹{mine_val:,.0f}")
        b.metric("Equal weight (lump sum, final)", f"₹{equal_val:,.0f}")
        (st.success if mine_val > equal_val else st.warning)(
            "The suggestion beat equal weight." if mine_val > equal_val else
            "The suggestion did NOT beat equal weight. That is a valid result: "
            "minimum variance targets lower risk, not higher return.")


def apply_split(pct):
    """Button callback: move the sidebar sliders to the optimised split."""
    for m in MARKETS:
        st.session_state[f"alloc_{m}"] = int(pct[m])


def render_optimiser(data, cfg):
    st.caption("Searches for the split that best meets your goal using only "
               "past data up to the end date. It is fitted to history, so "
               "check the walk-forward test below before trusting it.")
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox("Goal", OBJECTIVES)
    max_w = c2.slider("Max in any one market (%)", 40, 100, 70) / 100
    min_india = c3.slider("Min in India (%)", 0, 60, 20) / 100
    bounds = bounds_for(0.0, max_w, min_india)
    hist = data.loc[:pd.Timestamp(cfg["end"])]

    try:
        weights, (ret, vol, shp) = optimise(hist, objective, cfg["lookback"],
                                            cfg["rf"], bounds)
    except RuntimeError as err:
        st.warning(str(err))
        return
    pct = round_to_percent(weights)

    cols = st.columns(6)
    for col, mk in zip(cols[:3], MARKETS):
        col.metric(mk, f"{pct[mk]}%")
    cols[3].metric("Past return (ann.)", f"{ret:.1%}")
    cols[4].metric("Past volatility", f"{vol:.1%}")
    cols[5].metric("Past Sharpe", f"{shp:.2f}")
    st.button("Apply to my allocation", on_click=apply_split, args=(pct,))

    frontier = efficient_frontier(hist, cfg["lookback"], cfg["rf"], bounds)
    fig = px.scatter(frontier, x="Volatility", y="Return", color="Sharpe",
                     color_continuous_scale="Viridis", opacity=0.6,
                     title=f"Possible splits, last {cfg['lookback']} years")
    mu, cov = history_stats(hist, cfg["lookback"])
    for name, w in [("Optimised", weights.values), ("Equal weight", np.ones(3) / 3)]:
        r, v, _ = portfolio_stats(w, mu, cov, cfg["rf"])
        fig.add_scatter(x=[v], y=[r], mode="markers+text", text=[name],
                        textposition="top center", showlegend=False,
                        marker=dict(size=14, color="#E8833A" if name == "Optimised" else "#8A8F98",
                                    line=dict(width=2, color="white")))
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      margin=dict(t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Walk-forward test: did re-optimising each January pay off?")
    growth, wlog = walk_forward(data.loc[:pd.Timestamp(cfg["end"])], cfg["start"],
                                cfg["end"], objective, cfg["lookback"], cfg["rf"],
                                cfg["amount"], cfg["cost"], bounds)
    st.caption(f"Each year's weights use only the {cfg['lookback']} years before "
               "it. Equal weights are used until that much history exists. "
               "Lump sum, gross of fund fees.")
    fig = px.line(growth, labels={"value": "₹", "date": "", "variable": ""},
                  color_discrete_map={"Auto-optimised": "#7A3FD1",
                                      "Equal weight": "#8A8F98",
                                      "100% Nifty": "#E8833A"})
    fig.update_layout(hovermode="x unified", legend_orientation="h",
                      legend_y=1.1, margin=dict(t=30, b=0))
    st.plotly_chart(fig, width="stretch")
    summary = summarise(growth, cfg["rf"])
    st.dataframe(summary.style.format("{:.2%}", subset=["CAGR", "Volatility",
                                                        "Max drawdown"])
                 .format("{:.2f}", subset=["Sharpe"])
                 .format("₹{:,.0f}", subset=["Final value"]), width="stretch")
    won = summary.loc["Auto-optimised", "Sharpe"] > summary.loc["Equal weight", "Sharpe"]
    (st.success if won else st.warning)(
        "Out of sample, the optimiser had a higher Sharpe than equal weight."
        if won else "Out of sample, the optimiser did NOT beat equal weight on "
        "Sharpe. Optimised weights are fitted to the past and often fail to "
        "repeat; that is a valid result.")
    with st.expander("Weights chosen each January"):
        st.dataframe(wlog.style.format("{:.0%}"), width="stretch")


def render_downloads(mine, table, prices, split, rets):
    sheets = {"Monthly results": mine, "Market comparison": table,
              "Currency split": split, "Correlation": rets.corr(),
              "Monthly INR prices": prices}
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download Excel (all tables)", to_excel(sheets),
                       "portfolio_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇ Download CSV (month by month)", mine.to_csv(),
                       "results.csv", "text/csv")


# ------------------------------------------------------------------ main --
def main():
    st.title("🌏 India vs USA vs South Korea — in rupees")
    st.caption("A historical backtest of Nifty 50, S&P 500 and KOSPI converted "
               "to INR. It shows what would have happened, not what will.")

    data = get_data()
    first, last = data.index[0].date(), data.index[-1].date()
    cfg = read_inputs(first, last)
    validate(cfg)

    prices = data.loc[pd.Timestamp(cfg["start"]):pd.Timestamp(cfg["end"])]
    if len(prices) < 13:
        st.warning("Pick at least 12 months.")
        st.stop()

    mine, nifty, equal = run_simulations(prices, cfg)
    label, ret = headline_return(mine, cfg)
    table, rets = market_table(prices, cfg["rf"])

    render_headline(mine, label, ret)
    render_growth(mine, nifty, equal)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Markets compared", "Currency effect", "Correlation", "Lower-risk split",
         "Auto-optimiser"])
    with tab1:
        render_markets(table, prices)
    with tab2:
        render_currency(prices)
    with tab3:
        render_correlation(rets)
    with tab4:
        render_suggestion(data, cfg)
    with tab5:
        render_optimiser(data, cfg)

    st.divider()
    render_downloads(mine, table, prices, currency_split(prices), rets)
    st.caption(f"Data: yfinance, {first:%b %Y} to {last:%b %Y}. Price indices "
               "exclude dividends (~1-2%/yr). Taxes and LRS/TCS not modelled. "
               "Educational project, not financial advice.")


main()
