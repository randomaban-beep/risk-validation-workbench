"""Page 5: Historical & Hypothetical Stress Testing."""

import streamlit as st
import pandas as pd
import plotly.express as px

from rvw.config import load_config
from rvw.data.fetcher import load_prices
from rvw.portfolio.portfolio import create_preset_portfolio
from rvw.stress.covariance_stress import stress_covariance_matrix
from rvw.stress.historical_scenarios import run_all_historical_scenarios
from rvw.stress.hypothetical import evaluate_hypothetical_scenario

st.set_page_config(page_title="Stress Testing", page_icon="💥", layout="wide")
st.title("💥 Scenario Analysis & Stress Testing")
st.caption("Stress results represent scenario analysis, NOT statistical backtesting.")

config = load_config("config/default.yaml")
bundle = load_prices(config)
port = create_preset_portfolio("equal_weight", config.portfolios)

tab1, tab2, tab3 = st.tabs(["🏛️ Historical Replay", "⚡ Hypothetical Shocks", "🔄 Covariance Stress"])

with tab1:
    st.subheader("Historical Scenario Replay")
    scenarios = run_all_historical_scenarios(bundle.returns, port, config.scenarios)
    if scenarios:
        rows = [s.to_dict() for s in scenarios]
        st.dataframe(pd.DataFrame(rows)[["scenario_name", "start_date", "end_date", "trading_days", "cumulative_loss", "max_drawdown"]], use_container_width=True)

        selected_scen = st.selectbox("View Asset Performance Under Scenario:", [s.scenario_name for s in scenarios])
        target_s = next(s for s in scenarios if s.scenario_name == selected_scen)

        df_assets = pd.DataFrame(list(target_s.asset_cumulative_returns.items()), columns=["Asset", "Cumulative_Return"])
        fig = px.bar(df_assets, x="Asset", y="Cumulative_Return", color="Cumulative_Return", title=f"Asset Returns During: {selected_scen}", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Hypothetical Multi-Asset Shock & Duration Approx")
    st.markdown("*Illustrative assumptions, not calibrated forecasts.*")

    col1, col2 = st.columns(2)
    with col1:
        eq_shock = st.slider("Equities Shock (%)", -50.0, 10.0, -20.0, 5.0) / 100.0
        cr_shock = st.slider("Credit (HYG) Shock (%)", -30.0, 10.0, -8.0, 2.0) / 100.0
        gold_shock = st.slider("Gold (GLD) Shock (%)", -20.0, 30.0, 5.0, 5.0) / 100.0
    with col2:
        rates_bps = st.slider("Interest Rate Shift (bps)", -200, 300, 100, 25)

    hyp_res = evaluate_hypothetical_scenario(
        portfolio=port,
        scenario_id="custom_hyp",
        scenario_name="Interactive Custom Shock",
        asset_shocks={"SPY": eq_shock, "QQQ": eq_shock, "IWM": eq_shock, "HYG": cr_shock, "GLD": gold_shock},
        yield_shocks_bps={"TLT": rates_bps, "IEF": rates_bps},
        duration_assumptions={"TLT": 17.0, "IEF": 7.5},
    )

    st.metric("Total Portfolio Loss", f"{hyp_res.portfolio_loss:.2%}")

    df_contrib = pd.DataFrame(list(hyp_res.asset_loss_contributions.items()), columns=["Asset", "Loss_Contribution"])
    fig_h = px.bar(df_contrib, x="Asset", y="Loss_Contribution", title="Asset Loss Contributions under Shock")
    st.plotly_chart(fig_h, use_container_width=True)

with tab3:
    st.subheader("Covariance & Volatility Stress")
    vol_scale = st.slider("Asset Volatility Multiplier", 1.0, 3.0, 1.5, 0.1)
    corr_shift = st.slider("Correlation Shift towards +1.0", 0.0, 0.8, 0.3, 0.05)

    cov_res = stress_covariance_matrix(bundle.returns.iloc[-500:], port, vol_scale_factor=vol_scale, correlation_shift=corr_shift)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Baseline 99% VaR", f"{cov_res.baseline_var_99:.2%}")
    with c2:
        st.metric("Stressed 99% VaR", f"{cov_res.stressed_var_99:.2%}", delta=f"{cov_res.vol_increase_pct:.1f}% Vol Increase")
    with c3:
        st.metric("Stressed 99% ES", f"{cov_res.stressed_es_99:.2%}")

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
