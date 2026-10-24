"""End-to-end execution pipeline for model benchmarking, tuning, and backtesting."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from rvw.backtests.basel import compute_rolling_traffic_light
from rvw.backtests.suite import compare_models, evaluate_backtests
from rvw.config import AppConfig, load_config
from rvw.data.cache import save_cache
from rvw.data.fetcher import DataBundle, load_prices
from rvw.models import get_all_models
from rvw.portfolio.periods import PeriodSplit
from rvw.portfolio.portfolio import Portfolio, create_preset_portfolio
from rvw.tuning import freeze_hyperparameters, select_best_hyperparameters
from rvw.walkforward.engine import ForecastFrame, run_walkforward

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str = "config/default.yaml",
    force_synthetic: bool = False,
    refresh: bool = False,
    portfolio_name: str = "equal_weight",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Execute complete validation lifecycle: data -> tuning -> freeze -> test -> backtests -> artifacts.

    Args:
        config_path: Path to configuration file.
        force_synthetic: Force use of calibrated synthetic data.
        refresh: Force cache refresh.
        portfolio_name: Name of preset portfolio to evaluate.
        fast_mode: If True, uses reduced simulation count and higher refit_every for fast execution.

    Returns:
        Dict[str, Any]: Consolidated pipeline results and file paths.
    """
    config = load_config(config_path)
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest market data
    logger.info("Ingesting market data (force_synthetic=%s)...", force_synthetic)
    bundle = load_prices(config, refresh=refresh, force_synthetic=force_synthetic)
    returns = bundle.returns

    # 2. Portfolio setup
    portfolio = create_preset_portfolio(portfolio_name, config.portfolios)

    # 3. Period splitting
    period_split = PeriodSplit(returns, config)

    # Warm-up + validation data for tuning
    train_and_val = period_split.slice_data(
        config.periods.training.start_date,
        config.periods.validation.end_date,
    )

    # 4. Hyperparameter Selection on Validation
    w_cands = [250, 500] if fast_mode else config.walkforward.window_candidates
    lam_cands = config.models.ewma.get("lambda_candidates", [0.94, 0.97])

    tuning_res = select_best_hyperparameters(
        validation_returns=train_and_val,
        portfolio=portfolio,
        window_candidates=w_cands,
        lambda_candidates=lam_cands,
        var_alpha=0.99,
    )

    # Freeze hyperparameters
    frozen_path = freeze_hyperparameters(tuning_res, config)
    config.freeze_hyperparameters = True
    config.frozen_params_file = str(frozen_path)

    # Update period_split config reference
    period_split.config = config

    # Optimal window and parameters
    opt_window = tuning_res["selected_window_length"]
    opt_lambda = tuning_res["selected_ewma_lambda"]
    config.walkforward.window_length = opt_window
    config.models.ewma["lambda_param"] = opt_lambda

    if fast_mode:
        config.models.monte_carlo["num_simulations"] = 2000
        config.models.garch_fhs["resample_size"] = 1000
        refit_freq = 10
    else:
        refit_freq = config.walkforward.refit_every

    models = get_all_models(config)

    # 5. Evaluate Validation Period (2015-01-01 to 2018-12-31)
    val_sample = period_split.slice_data("2013-01-01", config.periods.validation.end_date)
    val_frames: List[ForecastFrame] = []
    val_summaries = []

    logger.info("Running walk-forward estimation on Validation period...")
    for m_name, model in models.items():
        logger.info("Evaluating model: %s on validation...", m_name)
        ff = run_walkforward(
            returns=val_sample,
            portfolio=portfolio,
            model=model,
            window_length=opt_window,
            refit_every=refit_freq,
            seed=config.seeds.global_seed,
        )
        val_frames.append(ff)
        summary = evaluate_backtests(ff, var_alpha=0.99, es_alpha=0.99)
        val_summaries.append(summary.to_dict())

    val_df = pd.DataFrame(val_summaries).set_index("model_name")
    val_df.to_parquet(results_dir / "validation_summary.parquet")

    # 6. Evaluate Test Period (2019-01-01 to end) - strictly guarded by freeze
    logger.info("Accessing test period under freeze guard...")
    # Slicing from warm-up prior to test period start
    test_full = period_split.slice_data("2017-01-01", config.periods.test.end_date)
    # Audited access to verify freeze enforcement
    _ = period_split.get_test(caller_context="pipeline_execution")

    test_frames: List[ForecastFrame] = []
    test_summaries = []

    logger.info("Running walk-forward estimation on Test period...")
    for m_name, model in models.items():
        logger.info("Evaluating model: %s on test set...", m_name)
        ff = run_walkforward(
            returns=test_full,
            portfolio=portfolio,
            model=model,
            window_length=opt_window,
            refit_every=refit_freq,
            seed=config.seeds.global_seed,
        )
        test_frames.append(ff)
        summary = evaluate_backtests(ff, var_alpha=0.99, es_alpha=0.99)
        test_summaries.append(summary.to_dict())

    test_df = pd.DataFrame(test_summaries).set_index("model_name")
    test_df.to_parquet(results_dir / "test_summary.parquet")

    # 7. Traffic light time series for benchmark model
    hs_frame = next(ff for ff in test_frames if ff.model_name == "historical_simulation")
    traffic_light_df = compute_rolling_traffic_light(hs_frame.hits["99"], window=250)
    traffic_light_df.to_parquet(results_dir / "traffic_light_series.parquet")

    # 8. Save forecast frames for reporting & dashboard
    combined_test_forecasts = pd.DataFrame(index=hs_frame.dates)
    combined_test_forecasts["Return"] = hs_frame.portfolio_returns.values
    combined_test_forecasts["Loss"] = hs_frame.realized_losses.values
    for ff in test_frames:
        if "99" in ff.var_forecasts.columns:
            combined_test_forecasts[f"{ff.model_name}_VaR_99"] = ff.var_forecasts["99"].values
        if "99" in ff.es_forecasts.columns:
            combined_test_forecasts[f"{ff.model_name}_ES_99"] = ff.es_forecasts["99"].values

    combined_test_forecasts.to_parquet(results_dir / "test_forecasts.parquet")

    # 9. Output results/summary.json and RESULTS_SUMMARY.md
    summary_data = {
        "execution_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "is_synthetic": bundle.is_synthetic,
        "source": bundle.source,
        "portfolio": portfolio.name,
        "tickers": list(portfolio.tickers),
        "test_evaluations_count": period_split.test_evaluation_count,
        "hyperparameter_selection": tuning_res,
        "validation_metrics": val_df.to_dict(orient="index"),
        "test_metrics": test_df.to_dict(orient="index"),
        "disclaimer": (
            "Educational prototype, inspired by model-validation practice; "
            "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
        ),
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    write_results_summary_markdown(summary_data, Path("RESULTS_SUMMARY.md"))

    return summary_data


def write_results_summary_markdown(summary_data: Dict[str, Any], output_path: Path) -> None:
    """Generate clean headline RESULTS_SUMMARY.md document."""
    test_metrics = summary_data["test_metrics"]
    is_synth = summary_data["is_synthetic"]
    synth_badge = "**DATA SOURCE: SYNTHETIC**" if is_synth else "**DATA SOURCE: REAL MARKET DATA (yfinance)**"

    md = f"""# Market Risk Model Validation Benchmark: Headline Results

{synth_badge}
*Run Timestamp*: {summary_data['execution_utc']}
*Portfolio*: `{summary_data['portfolio']}` ({len(summary_data['tickers'])} assets)
*Test Evaluations Count*: {summary_data['test_evaluations_count']}

> **Disclaimer**: {summary_data['disclaimer']}

---

## Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | CC p-val | Basel Zone | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for model_name, m in test_metrics.items():
        md += (
            f"| **{model_name}** | {m['observations']} | {m['exceptions']} | "
            f"{m['exception_rate']:.2%} | {m['expected_exceptions']} | {m['kupiec_p']:.4f} | "
            f"{m['conditional_coverage_p']:.4f} | **{m['basel_zone']}** | "
            f"{m['mcneil_frey_p']:.4f} | {m['mean_var']:.4f} | {m['mean_es']:.4f} |\n"
        )

    md += """
---

## Hyperparameter Selection (Validation Period Grid Search)
"""
    hp = summary_data["hyperparameter_selection"]
    md += f"- **Selected Rolling Window**: `{hp['selected_window_length']}` trading days\n"
    md += f"- **Selected EWMA Lambda**: `{hp['selected_ewma_lambda']}`\n"
    md += f"- **Objective Criterion**: `{hp['selection_metric']}` (Best score: `{hp['best_score']}`)\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
