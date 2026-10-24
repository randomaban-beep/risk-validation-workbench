"""Smoke and validation tests for Streamlit dashboard application files."""

import ast
from pathlib import Path
import pytest


def test_streamlit_files_compile_and_have_disclaimer():
    """Verify that all streamlit pages compile cleanly and include the mandatory disclaimer."""
    app_dir = Path("app")
    app_files = list(app_dir.glob("*.py")) + list((app_dir / "pages").glob("*.py"))
    assert len(app_files) == 8  # 1 main + 7 pages

    required_phrase = "educational prototype, inspired by model-validation practice"

    for py_file in app_files:
        content = py_file.read_text(encoding="utf-8")
        # Ensure python file compiles without syntax errors
        ast.parse(content, filename=str(py_file))
        # Ensure disclaimer is present in every page
        assert required_phrase.lower() in content.lower(), f"Missing disclaimer in {py_file}"
