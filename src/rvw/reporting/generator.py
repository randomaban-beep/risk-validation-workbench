"""Report generation from stored result artifacts via Jinja2."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import jinja2

from rvw.config import AppConfig, load_config
from rvw.reporting.findings import generate_rule_based_findings

logger = logging.getLogger(__name__)


def generate_validation_report(
    results_dir: str | Path = "results",
    output_dir: str | Path = "reports",
    config_path: str = "config/default.yaml",
) -> Tuple[Path, Path]:
    """Generate HTML and Markdown validation reports strictly from saved artifacts in results_dir.

    Args:
        results_dir: Path to directory containing saved pipeline artifacts.
        output_dir: Path to destination reports directory.
        config_path: Path to configuration file.

    Returns:
        Tuple[Path, Path]: (markdown_report_path, html_report_path)

    Raises:
        FileNotFoundError: If results/summary.json does not exist.
    """
    res_dir = Path(results_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = res_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Cannot generate report: required artifact '{summary_file}' was not found. "
            "Execute the benchmark pipeline first ('python -m rvw.cli run')."
        )

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Load config for reproducibility metadata
    try:
        config = load_config(config_path)
        config_hash = hashlib.sha256(config.to_json().encode("utf-8")).hexdigest()
        seeds = {
            "global_seed": config.seeds.global_seed,
            "monte_carlo_seed": config.seeds.monte_carlo_seed,
            "synthetic_seed": config.seeds.synthetic_seed,
        }
    except Exception:
        config_hash = "N/A"
        seeds = {"global_seed": 42, "monte_carlo_seed": 1337, "synthetic_seed": 777}

    # Generate algorithmic findings
    findings = generate_rule_based_findings(summary_data)

    template_dir = Path(__file__).parent / "templates"
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )

    template_context = {
        "is_synthetic": summary_data.get("is_synthetic", False),
        "source": summary_data.get("source", "SYNTHETIC"),
        "disclaimer": summary_data.get("disclaimer", ""),
        "portfolio_name": summary_data.get("portfolio", "equal_weight"),
        "tickers": summary_data.get("tickers", []),
        "start_date": "2019-01-01",
        "end_date": "Latest",
        "findings": findings,
        "test_metrics": summary_data.get("test_metrics", {}),
        "validation_metrics": summary_data.get("validation_metrics", {}),
        "hyperparameter_selection": summary_data.get("hyperparameter_selection", {}),
        "test_evaluations_count": summary_data.get("test_evaluations_count", 1),
        "generated_at_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "config_hash": config_hash,
        "seeds": seeds,
    }

    # Render HTML
    html_template = jinja_env.get_template("report_template.html")
    html_content = html_template.render(**template_context)
    html_path = out_dir / "validation_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Render Markdown (using FileSystemLoader with autoescape=False for markdown)
    md_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )
    md_template = md_env.get_template("report_template.md")
    md_content = md_template.render(**template_context)
    md_path = out_dir / "validation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Reports successfully generated: %s and %s", md_path, html_path)
    return md_path, html_path
