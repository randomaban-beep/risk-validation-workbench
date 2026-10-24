"""Market Risk Model Validation Workbench - Interactive Streamlit Dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from rvw.config import load_config

# Page configuration
st.set_page_config(
    page_title="Market Risk Model Validation Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Market Risk Model Validation Workbench")
st.caption("Quantitative Benchmarking, Backtesting, Stress Testing & Risk Attribution Framework")

# Load configuration and summary data
config = load_config("config/default.yaml")
summary_file = Path(config.paths.results_dir) / "summary.json"

if summary_file.exists():
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
else:
    summary_data = {}

# Sidebar status
with st.sidebar:
    st.header("Workbench Status")
    if summary_data.get("is_synthetic", False):
        st.warning("⚠️ Data Mode: **SYNTHETIC**")
    else:
        st.success(f"✅ Data Mode: **{summary_data.get('source', 'Online')}**")

    st.markdown(f"**Indicative Assets**: {len(config.universe.tickers)}")
    st.markdown(f"**Horizon**: {config.risk_conventions.horizon_days} Trading Day")
    st.markdown(f"**VaR Alphas**: {config.risk_conventions.var_alphas}")
    st.markdown(f"**ES Alphas**: {config.risk_conventions.es_alphas}")
    st.divider()
    st.info("Navigate through the validation workflow using the pages menu above.")

# Main Dashboard Overview
col1, col2, col3, col4 = st.columns(4)
test_metrics = summary_data.get("test_metrics", {})
hs_metrics = test_metrics.get("historical_simulation", {})

with col1:
    st.metric("Test Observations", f"{hs_metrics.get('observations', 0):,}")
with col2:
    st.metric("Benchmark Exceptions", f"{hs_metrics.get('exceptions', 0)} ({hs_metrics.get('exception_rate', 0.0):.2%})")
with col3:
    st.metric("Expected Exceptions", f"{hs_metrics.get('expected_exceptions', 0.0):.1f}")
with col4:
    st.metric("Basel Traffic Light", hs_metrics.get("basel_zone", "N/A"))

st.subheader("Model Validation Summary Table (99% VaR & 99% ES)")
if test_metrics:
    df_metrics = pd.DataFrame.from_dict(test_metrics, orient="index")
    # Clean up display columns
    display_cols = [
        "observations", "exceptions", "exception_rate", "expected_exceptions",
        "kupiec_p", "conditional_coverage_p", "basel_zone", "mcneil_frey_p",
        "mean_var", "mean_es"
    ]
    avail_cols = [c for c in display_cols if c in df_metrics.columns]
    st.dataframe(df_metrics[avail_cols], use_container_width=True)
else:
    st.warning("Pipeline results artifact not found. Run `python -m rvw.cli run` to generate benchmark data.")

# Quick links to modules
st.subheader("Workbench Capabilities")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🔍 Data Quality")
    st.write("Examine data integrity audits, missing value reports, stale pricing checks, and jump alerts.")
with c2:
    st.markdown("### 📊 Backtest Suite")
    st.write("Kupiec POF, Christoffersen Independence, Conditional Coverage, and Basel 250-day Traffic Light.")
with c3:
    st.markdown("### 💥 Stress & Attribution")
    st.write("Historical replay (GFC, COVID), duration shocks, covariance stress, and Euler risk attribution.")

# Footer with mandatory disclaimer
st.markdown("---")
st.markdown(
    "<small style='color: grey;'>"
    "<strong>Regulatory & Production Disclaimer:</strong> "
    "Educational prototype, inspired by model-validation practice; "
    "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
    "</small>",
    unsafe_allow_html=True,
)
