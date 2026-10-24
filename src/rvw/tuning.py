"""Hyperparameter grid search, model selection, and freeze protocol."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from rvw.backtests.suite import evaluate_backtests
from rvw.config import AppConfig
from rvw.models.ewma import EWMAModel
from rvw.models.historical import HistoricalSimulationModel
from rvw.models.parametric_normal import ParametricNormalModel
from rvw.models.parametric_t import ParametricStudentTModel
from rvw.portfolio.periods import PeriodSplit
from rvw.portfolio.portfolio import Portfolio
from rvw.walkforward.engine import ForecastFrame, run_walkforward

logger = logging.getLogger(__name__)


def select_best_hyperparameters(
    validation_returns: pd.DataFrame,
    portfolio: Portfolio,
    window_candidates: List[int],
    lambda_candidates: List[float],
    var_alpha: float = 0.99,
) -> Dict[str, Any]:
    """Execute hyperparameter grid search on the Validation period.

    Evaluates candidate window lengths [250, 500, 1000] and EWMA decay factors [0.94, 0.97]
    against the 99% VaR conditional coverage objective.

    Selection criterion:
    Minimizes score = |actual_rate - 0.01| * 100 + LR_cc_stat * 0.5

    Args:
        validation_returns: Return series for validation window (includes warm-up).
        portfolio: Target portfolio.
        window_candidates: List of candidate window lengths.
        lambda_candidates: Candidate EWMA lambda parameters.
        var_alpha: Confidence level for evaluation (0.99).

    Returns:
        Dict[str, Any]: Selected optimal hyperparameters and full audit trail.
    """
    logger.info("Initiating hyperparameter selection on validation period...")
    records = []

    best_score = float("inf")
    best_window = window_candidates[0]
    best_lambda = lambda_candidates[0]

    for w in window_candidates:
        if len(validation_returns) <= w + 10:
            continue
        # Test Historical Simulation and Parametric Normal on this window
        for model in [HistoricalSimulationModel(), ParametricNormalModel()]:
            ff = run_walkforward(
                returns=validation_returns,
                portfolio=portfolio,
                model=model,
                window_length=w,
                refit_every=1,
            )
            summary = evaluate_backtests(ff, var_alpha=var_alpha)
            score = abs(summary.exception_rate - (1.0 - var_alpha)) * 100.0 + summary.conditional_coverage_lr * 0.5
            records.append({
                "model": model.name,
                "window": w,
                "lambda": None,
                "score": score,
                "exceptions": summary.exceptions,
                "p_cc": summary.conditional_coverage_p,
            })
            if score < best_score:
                best_score = score
                best_window = w

        # Test EWMA across candidate lambdas
        for lam in lambda_candidates:
            ewma_m = EWMAModel(params={"lambda_param": lam})
            ff = run_walkforward(
                returns=validation_returns,
                portfolio=portfolio,
                model=ewma_m,
                window_length=w,
                refit_every=1,
            )
            summary = evaluate_backtests(ff, var_alpha=var_alpha)
            score = abs(summary.exception_rate - (1.0 - var_alpha)) * 100.0 + summary.conditional_coverage_lr * 0.5
            records.append({
                "model": "ewma",
                "window": w,
                "lambda": lam,
                "score": score,
                "exceptions": summary.exceptions,
                "p_cc": summary.conditional_coverage_p,
            })
            if score < best_score:
                best_score = score
                best_window = w
                best_lambda = lam

    return {
        "selected_window_length": best_window,
        "selected_ewma_lambda": best_lambda,
        "selection_metric": "min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)",
        "best_score": round(best_score, 4),
        "grid_evaluations": records,
    }


def freeze_hyperparameters(
    selected_params: Dict[str, Any],
    config: AppConfig,
    destination_path: Optional[str | Path] = None,
) -> Path:
    """Save selected hyperparameters to frozen JSON artifact.

    Args:
        selected_params: Dictionary of chosen parameters.
        config: Application configuration.
        destination_path: Target path (defaults to config.frozen_params_file).

    Returns:
        Path: Path to written frozen artifact.
    """
    dest = Path(destination_path or config.frozen_params_file)
    dest.parent.mkdir(parents=True, exist_ok=True)

    config_str = config.to_json()
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    payload = {
        "status": "FROZEN",
        "frozen_at_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "selected_hyperparameters": selected_params,
        "universe": config.universe.tickers,
        "portfolios": list(config.portfolios.keys()),
        "disclaimer": (
            "Educational prototype, inspired by model-validation practice; "
            "not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."
        ),
    }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Hyperparameters frozen and locked at: %s", dest.resolve())
    return dest
