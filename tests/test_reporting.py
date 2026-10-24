"""Unit tests for artifact-driven report generation and findings extraction."""

import json
import re
from pathlib import Path
import pytest

from rvw.reporting.findings import generate_rule_based_findings
from rvw.reporting.generator import generate_validation_report


_ABSOLUTE_PATH_PATTERN = re.compile(
    r"([A-Za-z]:\\Users|/home/[^/]+|[A-Za-z]:\\\\Users)",
    re.IGNORECASE,
)


def test_rule_based_findings_extraction():
    """Verify algorithmic findings extraction from backtest summary data."""
    mock_summary = {
        "is_synthetic": True,
        "source": "SYNTHETIC",
        "portfolio": "equal_weight",
        "tickers": ["SPY", "TLT"],
        "hyperparameter_selection": {
            "selected_window_length": 500,
            "selected_ewma_lambda": 0.94,
            "selection_metric": "min(|actual_rate - 0.01|)",
            "best_score": 0.05,
        },
        "test_metrics": {
            "historical_simulation": {
                "observations": 500,
                "exceptions": 5,
                "expected_exceptions": 5.0,
                "exception_rate": 0.01,
                "kupiec_p": 0.95,
                "independence_p": 0.80,
                "conditional_coverage_p": 0.90,
                "basel_zone": "Green",
                "mcneil_frey_p": 0.45,
            },
            "parametric_normal": {
                "observations": 500,
                "exceptions": 18,
                "expected_exceptions": 5.0,
                "exception_rate": 0.036,
                "kupiec_p": 0.001,
                "independence_p": 0.02,
                "conditional_coverage_p": 0.0005,
                "basel_zone": "Red",
                "mcneil_frey_p": 0.01,
            },
        },
    }

    findings = generate_rule_based_findings(mock_summary)
    text = " ".join(findings)

    assert "SYNTHETIC" in text
    assert "Green zone" in text
    assert "Red zone" in text
    assert "Kupiec Unconditional Coverage rejection" in text
    assert "Christoffersen Independence rejection" in text
    assert "McNeil-Frey ES Exceedance test rejection" in text


def test_generate_validation_report_artifacts(tmp_path: Path):
    """Verify HTML and Markdown report generation from results artifacts."""
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()

    mock_summary = {
        "execution_utc": "2026-10-24T12:00:00Z",
        "is_synthetic": True,
        "source": "SYNTHETIC",
        "portfolio": "equal_weight",
        "tickers": ["SPY", "TLT"],
        "test_evaluations_count": 1,
        "disclaimer": (
            "Educational prototype, inspired by model-validation practice; "
            "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
        ),
        "hyperparameter_selection": {
            "selected_window_length": 250,
            "selected_ewma_lambda": 0.94,
            "selection_metric": "min score",
            "best_score": 0.1,
        },
        "test_metrics": {
            "historical_simulation": {
                "observations": 250,
                "exceptions": 2,
                "expected_exceptions": 2.5,
                "exception_rate": 0.008,
                "kupiec_p": 0.74,
                "independence_p": 0.90,
                "conditional_coverage_p": 0.85,
                "basel_zone": "Green",
                "mcneil_frey_p": 0.50,
                "mean_var": 0.021,
                "mean_es": 0.027,
            }
        },
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    md_path, html_path = generate_validation_report(
        results_dir=results_dir,
        output_dir=reports_dir,
        config_path="config/synthetic_test.yaml",
    )

    assert md_path.exists()
    assert html_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")

    # Verify mandatory regulatory disclaimer is present in both
    disclaimer_snippet = "educational prototype, inspired by model-validation practice"
    assert disclaimer_snippet.lower() in md_text.lower()
    assert disclaimer_snippet.lower() in html_text.lower()

    # Verify numbers from summary.json match report content
    assert "0.0210" in md_text or "0.021" in md_text
    assert "Green" in md_text


def test_report_contains_no_absolute_local_paths(tmp_path: Path):
    """Verify generated reports contain no absolute local file-system paths (e.g. C:\\Users\\...).

    Absolute paths embedded in reports would be non-reproducible on other machines and would
    expose local directory structure unnecessarily.
    """
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()

    mock_summary = {
        "execution_utc": "2026-10-24T12:00:00Z",
        "is_synthetic": True,
        "source": "SYNTHETIC",
        "portfolio": "equal_weight",
        "tickers": ["SPY", "TLT"],
        "test_evaluations_count": 1,
        "disclaimer": (
            "Educational prototype, inspired by model-validation practice; "
            "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
        ),
        "hyperparameter_selection": {
            "selected_window_length": 250,
            "selected_ewma_lambda": 0.94,
            "selection_metric": "min score",
            "best_score": 0.1,
        },
        "test_metrics": {
            "historical_simulation": {
                "observations": 250,
                "exceptions": 2,
                "expected_exceptions": 2.5,
                "exception_rate": 0.008,
                "kupiec_p": 0.74,
                "independence_p": 0.90,
                "conditional_coverage_p": 0.85,
                "basel_zone": "Green",
                "basel_zone_worst": "Green",
                "basel_zone_latest": "Green",
                "mcneil_frey_p": 0.50,
                "mean_var": 0.021,
                "mean_es": 0.027,
            }
        },
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    md_path, html_path = generate_validation_report(
        results_dir=results_dir,
        output_dir=reports_dir,
        config_path="config/synthetic_test.yaml",
    )

    for report_path in [md_path, html_path]:
        content = report_path.read_text(encoding="utf-8")
        matches = _ABSOLUTE_PATH_PATTERN.findall(content)
        assert not matches, (
            f"Absolute local path found in {report_path.name}: {matches}. "
            "Reports must not contain machine-specific absolute paths."
        )
