"""Command-Line Interface (CLI) for the Market Risk Model Validation Workbench."""

from __future__ import annotations

import logging
from pathlib import Path
import click

from rvw.config import load_config
from rvw.data.fetcher import load_prices
from rvw.logging_utils import setup_logging
from rvw.pipeline import run_pipeline
from rvw.reporting.generator import generate_validation_report

logger = logging.getLogger("rvw.cli")


@click.group()
def cli() -> None:
    """Market Risk Model Validation Workbench CLI."""
    pass


@cli.command("fetch-data")
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML.")
@click.option("--refresh", is_flag=True, default=False, help="Force refresh and redownload.")
@click.option("--synthetic", is_flag=True, default=False, help="Force generation of synthetic data.")
def fetch_data_cmd(config_path: str, refresh: bool, synthetic: bool) -> None:
    """Fetch or generate market price data and build cache."""
    setup_logging()
    cfg = load_config(config_path)
    click.echo(f"Ingesting data for {len(cfg.universe.tickers)} tickers...")
    bundle = load_prices(cfg, refresh=refresh, force_synthetic=synthetic)
    click.echo(f"Successfully processed {len(bundle.prices)} rows.")
    click.echo(f"Data source: {bundle.source} (is_synthetic={bundle.is_synthetic})")
    click.echo(f"Quality report valid: {bundle.quality_report.is_valid} ({len(bundle.quality_report.issues)} issues logged)")


@cli.command("run")
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML.")
@click.option("--synthetic", is_flag=True, default=False, help="Force synthetic data.")
@click.option("--refresh", is_flag=True, default=False, help="Refresh data cache.")
@click.option("--portfolio", "portfolio_name", default="equal_weight", help="Preset portfolio name.")
@click.option("--fast", is_flag=True, default=False, help="Fast execution mode with reduced simulation paths.")
def run_cmd(config_path: str, synthetic: bool, refresh: bool, portfolio_name: str, fast: bool) -> None:
    """Execute complete validation pipeline: tuning, freeze, test, and backtests."""
    setup_logging()
    click.echo(f"Starting validation pipeline for portfolio '{portfolio_name}'...")
    summary = run_pipeline(
        config_path=config_path,
        force_synthetic=synthetic,
        refresh=refresh,
        portfolio_name=portfolio_name,
        fast_mode=fast,
    )
    click.echo("Validation pipeline completed successfully!")
    click.echo(f"Results summary written to results/summary.json and RESULTS_SUMMARY.md")


@cli.command("report")
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML.")
@click.option("--results-dir", default="results", help="Directory containing pipeline artifacts.")
@click.option("--output-dir", default="reports", help="Output directory for generated reports.")
def report_cmd(config_path: str, results_dir: str, output_dir: str) -> None:
    """Generate HTML and Markdown reports from saved result artifacts."""
    setup_logging()
    click.echo(f"Generating reports from '{results_dir}' to '{output_dir}'...")
    md_path, html_path = generate_validation_report(
        results_dir=results_dir,
        output_dir=output_dir,
        config_path=config_path,
    )
    click.echo(f"Markdown report generated: {md_path}")
    click.echo(f"HTML report generated:     {html_path}")


def main() -> None:
    """Entry point for CLI execution."""
    cli()


if __name__ == "__main__":
    main()
