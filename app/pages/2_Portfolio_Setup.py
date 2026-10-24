"""Page 2: Portfolio Builder and Preset Allocation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from rvw.config import load_config
from rvw.portfolio.portfolio import Portfolio, create_preset_portfolio

st.set_page_config(page_title="Portfolio Setup", page_icon="💼", layout="wide")
st.title("💼 Portfolio Setup & Asset Allocations")

config = load_config("config/default.yaml")
tickers = config.universe.tickers

preset_options = list(config.portfolios.keys()) + ["Custom"]
selected_preset = st.selectbox("Select Portfolio Preset:", preset_options, index=0)

if selected_preset != "Custom":
    preset_weights = config.portfolios[selected_preset]
    port = create_preset_portfolio(selected_preset, config.portfolios)
    st.success(f"Preset **{selected_preset}** loaded ({len(port.tickers)} active assets).")
    weights_dict = port.to_dict()
else:
    st.write("Customize asset allocation weights (must sum to 100%):")
    cols = st.columns(4)
    weights_dict = {}
    default_w = round(100.0 / len(tickers), 2)
    for idx, t in enumerate(tickers):
        with cols[idx % 4]:
            val = st.number_input(f"{t} Weight (%)", min_value=0.0, max_value=100.0, value=default_w, step=1.0)
            weights_dict[t] = val / 100.0

total_pct = sum(weights_dict.values()) * 100.0
st.write(f"**Total Allocation:** {total_pct:.2f}%")

if not np.isclose(total_pct, 100.0, atol=0.1):
    st.error(f"Total weights must equal 100%. Currently: {total_pct:.2f}%.")
else:
    st.success("Weights are valid and normalized to 1.0.")
    active_weights = {k: v for k, v in weights_dict.items() if v > 0.0}
    df_w = pd.DataFrame(list(active_weights.items()), columns=["Asset", "Weight"])
    fig = px.pie(df_w, values="Weight", names="Asset", title="Portfolio Asset Allocation", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
