"""Page 3: Model Comparison & VaR Envelopes."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from rvw.config import load_config

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("📈 Model Comparison & VaR Forecast Envelopes")

config = load_config("config/default.yaml")
forecast_path = Path(config.paths.results_dir) / "test_forecasts.parquet"

if not forecast_path.exists():
    st.warning("Forecast artifact not found. Please run the validation pipeline first: `python -m rvw.cli run`.")
else:
    df_fc = pd.read_parquet(forecast_path)
    model_keys = ["historical_simulation", "parametric_normal", "parametric_t", "ewma", "garch_fhs", "monte_carlo"]
    avail_models = [m for m in model_keys if f"{m}_VaR_99" in df_fc.columns]

    selected_model = st.selectbox("Select Model for Detailed Envelope:", avail_models, index=0)

    # Plot Out-of-Sample VaR and Realized Losses
    var_col = f"{selected_model}_VaR_99"
    es_col = f"{selected_model}_ES_99"

    fig = go.Figure()
    # Realized losses
    fig.add_trace(go.Scatter(
        x=df_fc.index,
        y=df_fc["Loss"],
        mode="lines",
        name="Realized Daily Loss",
        line=dict(color="rgba(100, 116, 139, 0.6)", width=1),
    ))
    # VaR 99%
    fig.add_trace(go.Scatter(
        x=df_fc.index,
        y=df_fc[var_col],
        mode="lines",
        name=f"99% VaR ({selected_model})",
        line=dict(color="#2563eb", width=2),
    ))
    # ES 99%
    if es_col in df_fc.columns:
        fig.add_trace(go.Scatter(
            x=df_fc.index,
            y=df_fc[es_col],
            mode="lines",
            name=f"99% Expected Shortfall",
            line=dict(color="#dc2626", width=1.5, dash="dash"),
        ))

    # Highlight Exceptions
    exceptions_mask = df_fc["Loss"] > df_fc[var_col]
    exc_dates = df_fc.index[exceptions_mask]
    exc_losses = df_fc["Loss"][exceptions_mask]

    fig.add_trace(go.Scatter(
        x=exc_dates,
        y=exc_losses,
        mode="markers",
        name=f"VaR Exceptions ({len(exc_dates)})",
        marker=dict(color="red", size=7, symbol="x"),
    ))

    fig.update_layout(
        title=f"Out-of-Sample 99% VaR & ES Envelope: {selected_model}",
        xaxis_title="Date",
        yaxis_title="Fractional Loss of Portfolio",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cross-model VaR comparison
    st.subheader("Cross-Model VaR 99% Forecasts Overlay")
    fig_comp = go.Figure()
    for m in avail_models:
        fig_comp.add_trace(go.Scatter(
            x=df_fc.index,
            y=df_fc[f"{m}_VaR_99"],
            mode="lines",
            name=m,
            line=dict(width=1.5),
        ))
    fig_comp.update_layout(
        title="Comparative 99% VaR Trajectories Across All Six Models",
        xaxis_title="Date",
        yaxis_title="99% Daily VaR",
        template="plotly_white",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
