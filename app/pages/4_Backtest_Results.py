"""Page 4: Regulatory and Statistical Backtest Suite."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rvw.config import load_config

st.set_page_config(page_title="Backtest Results", page_icon="🚦", layout="wide")
st.title("🚦 Statistical Backtest Suite & Basel Traffic Light")

config = load_config("config/default.yaml")
res_dir = Path(config.paths.results_dir)

test_summary_path = res_dir / "test_summary.parquet"
traffic_light_path = res_dir / "traffic_light_series.parquet"

if not test_summary_path.exists():
    st.warning("Backtest summary not found. Run `python -m rvw.cli run` first.")
else:
    df_test = pd.read_parquet(test_summary_path)

    st.subheader("Statistical Coverage and Hypothesis Test Summary (Test Period)")
    st.dataframe(df_test, use_container_width=True)

    # Explanation of hypothesis tests
    with st.expander("ℹ️ Explanation of Statistical Tests"):
        st.markdown(r"""
        - **Kupiec POF (Unconditional Coverage)**: Tests whether the total number of VaR exceptions matches expected frequency ($p_0 = 1 - \alpha$). $H_0$ is rejected if $p < 0.05$.
        - **Christoffersen Independence**: Tests whether exceptions occur independently or cluster in time (first-order Markov chain). Rejection indicates clustering.
        - **Christoffersen Conditional Coverage**: Joint likelihood ratio test of unconditional coverage and independence ($\text{LR}_{cc} = \text{LR}_{uc} + \text{LR}_{ind} \sim \chi^2(2)$).
        - **McNeil-Frey ES Exceedance**: Evaluates whether tail losses exceeding VaR systematically surpass Expected Shortfall ($H_0: \mathbb{E}[L - \text{ES} \mid L > \text{VaR}] = 0$).
        """)

    # Rolling Basel Traffic Light
    st.subheader("Rolling 250-Day Basel Traffic Light Classification (Historical Simulation)")
    if traffic_light_path.exists():
        tl_df = pd.read_parquet(traffic_light_path)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tl_df.index,
            y=tl_df["Rolling_Exceptions"],
            mode="lines+markers",
            name="Rolling 250-Day Exceptions",
            line=dict(color="#1e293b", width=2),
            marker=dict(size=4),
        ))

        # Horizontal regulatory boundaries
        fig.add_hline(y=4.5, line_dash="dash", line_color="green", annotation_text="Green / Yellow Boundary (4)")
        fig.add_hline(y=9.5, line_dash="dash", line_color="red", annotation_text="Yellow / Red Boundary (9)")

        fig.update_layout(
            title="Basel Framework Rolling 250-Day 99% VaR Exception Count",
            xaxis_title="Date",
            yaxis_title="Exceptions in Prior 250 Days",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rolling traffic light time series not yet generated.")

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
