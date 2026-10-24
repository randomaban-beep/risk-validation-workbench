"""Page 6: Euler Risk Attribution & What-If Rebalancing."""

import streamlit as st
import pandas as pd
import plotly.express as px

from rvw.attribution.euler import compute_euler_parametric_risk, compute_historical_tail_attribution
from rvw.attribution.what_if import compute_what_if_rebalance
from rvw.config import load_config
from rvw.data.fetcher import load_prices
from rvw.portfolio.portfolio import create_preset_portfolio

st.set_page_config(page_title="Risk Attribution", page_icon="🧩", layout="wide")
st.title("🧩 Euler Risk Attribution & What-If Rebalancing")

config = load_config("config/default.yaml")
bundle = load_prices(config)
returns_window = bundle.returns.iloc[-500:]

tab1, tab2 = st.tabs(["📊 Euler Component Risk", "⚖️ What-If Rebalancing"])

with tab1:
    selected_preset = st.selectbox("Select Portfolio for Decomposition:", list(config.portfolios.keys()), index=0)
    port = create_preset_portfolio(selected_preset, config.portfolios)
    alpha = st.selectbox("Confidence Level:", [0.99, 0.95], index=0)

    euler_res = compute_euler_parametric_risk(returns_window, port, alpha=alpha)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Parametric VaR", f"{euler_res.total_var:.2%}")
    with c2:
        st.metric("Total Expected Shortfall", f"{euler_res.total_es:.2%}")
    with c3:
        st.metric("Euler Sum Discrepancy", f"{euler_res.euler_sum_error_var:.1e}")

    st.subheader("Asset Risk Contributions Table")
    df_decomp = euler_res.to_dataframe()
    st.dataframe(df_decomp, use_container_width=True)

    fig = px.bar(
        df_decomp.reset_index(),
        x="Asset",
        y="Pct_VaR",
        color="Pct_VaR",
        title="Percentage Contribution to Total VaR (%)",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What-If Portfolio Rebalancing Comparator")
    col1, col2 = st.columns(2)
    with col1:
        curr_p_name = st.selectbox("Current Allocation:", list(config.portfolios.keys()), index=0)
    with col2:
        tgt_p_name = st.selectbox("Target Allocation:", list(config.portfolios.keys()), index=1)

    p_curr = create_preset_portfolio(curr_p_name, config.portfolios)
    p_tgt = create_preset_portfolio(tgt_p_name, config.portfolios)

    whatif = compute_what_if_rebalance(returns_window, p_curr, p_tgt, alpha=0.99)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Current 99% VaR", f"{whatif.current_var:.2%}")
    with m2:
        st.metric("Target 99% VaR", f"{whatif.target_var:.2%}", delta=f"{whatif.pct_delta_var:+.1f}%")
    with m3:
        st.metric("Delta Volatility", f"{whatif.delta_vol:+.2%}")

    df_deltas = pd.DataFrame(list(whatif.asset_cvar_deltas.items()), columns=["Asset", "Component_VaR_Delta"])
    fig_w = px.bar(df_deltas, x="Asset", y="Component_VaR_Delta", title="Change in Component VaR by Asset", color="Component_VaR_Delta", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_w, use_container_width=True)

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
