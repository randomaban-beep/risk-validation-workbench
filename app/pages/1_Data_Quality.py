"""Page 1: Data Ingestion and Quality Validation Audit."""

import streamlit as st
import pandas as pd
import plotly.express as px

from rvw.config import load_config
from rvw.data.fetcher import load_prices

st.set_page_config(page_title="Data Quality Audit", page_icon="🔍", layout="wide")
st.title("🔍 Data Quality Audit & Ingestion Integrity")

config = load_config("config/default.yaml")
bundle = load_prices(config, refresh=False, force_synthetic=False)
report = bundle.quality_report

# Header status
if bundle.is_synthetic:
    st.warning("⚠️ **DATA STATUS: SYNTHETIC**. Price paths generated via calibrated Student-t copula.")
else:
    st.success(f"✅ **DATA STATUS: {bundle.source}**. Real market data cached.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{report.total_rows:,}")
with col2:
    st.metric("Universe Assets", report.total_columns)
with col3:
    st.metric("Date Range", f"{report.start_date} to {report.end_date}")
with col4:
    st.metric("Integrity Check", "PASSED" if report.is_valid else "FAILED")

st.subheader("Asset-Level Quality Metrics")
summary_df = report.to_summary_df()
if not summary_df.empty:
    st.dataframe(summary_df, use_container_width=True)

st.subheader("Identified Data Quality Events")
if report.issues:
    issues_data = [{"Category": i.category, "Ticker": i.ticker, "Date": i.date, "Severity": i.severity, "Message": i.message} for i in report.issues]
    st.dataframe(pd.DataFrame(issues_data), use_container_width=True)
else:
    st.info("No critical data anomalies or calendar gaps detected.")

# Interactive price plot
st.subheader("Normalized Asset Price History (Base = 100)")
selected_tickers = st.multiselect("Select assets to display:", list(bundle.prices.columns), default=["SPY", "TLT", "GLD"])
if selected_tickers:
    norm_prices = bundle.prices[selected_tickers] / bundle.prices[selected_tickers].iloc[0] * 100.0
    fig = px.line(norm_prices, x=norm_prices.index, y=norm_prices.columns, title="Asset Price Trajectories (Normalized)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Indexed Value (Base 100)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
