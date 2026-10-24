"""Page 7: Report Generation and Export."""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from rvw.config import load_config
from rvw.reporting.generator import generate_validation_report

st.set_page_config(page_title="Report Export", page_icon="📄", layout="wide")
st.title("📄 Validation Report Export & Review")

config = load_config("config/default.yaml")
rep_dir = Path(config.paths.reports_dir)
html_path = rep_dir / "validation_report.html"
md_path = rep_dir / "validation_report.md"

if st.button("🔄 Regenerate Reports from Current Artifacts"):
    with st.spinner("Generating fresh reports..."):
        generate_validation_report(
            results_dir=config.paths.results_dir,
            output_dir=config.paths.reports_dir,
            config_path="config/default.yaml",
        )
    st.success("Reports regenerated successfully!")

col1, col2 = st.columns(2)
with col1:
    if html_path.exists():
        html_bytes = html_path.read_bytes()
        st.download_button(
            label="⬇️ Download HTML Report",
            data=html_bytes,
            file_name="validation_report.html",
            mime="text/html",
        )
with col2:
    if md_path.exists():
        md_bytes = md_path.read_bytes()
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=md_bytes,
            file_name="validation_report.md",
            mime="text/markdown",
        )

st.subheader("Interactive HTML Report Preview")
if html_path.exists():
    html_content = html_path.read_text(encoding="utf-8")
    components.html(html_content, height=800, scrolling=True)
else:
    st.info("HTML report does not exist yet. Click 'Regenerate Reports' or run `python -m rvw.cli report`.")

st.markdown("---")
st.markdown("<small style='color: grey;'>Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.</small>", unsafe_allow_html=True)
