"""Unit tests for hyperparameter selection, validation tuning, and freeze protocol."""

import json
from pathlib import Path
import pandas as pd
import pytest

from rvw.portfolio.portfolio import Portfolio
from rvw.tuning import freeze_hyperparameters, select_best_hyperparameters


def test_select_best_hyperparameters(sample_returns, synthetic_test_config):
    """Verify grid search returns valid hyperparameters on validation data."""
    port = Portfolio(weights=pd.Series({"SPY": 0.5, "TLT": 0.5}), name="port")
    val_sample = sample_returns.iloc[:400]

    res = select_best_hyperparameters(
        validation_returns=val_sample,
        portfolio=port,
        window_candidates=[100, 200],
        lambda_candidates=[0.94],
        var_alpha=0.99,
    )

    assert "selected_window_length" in res
    assert res["selected_window_length"] in [100, 200]
    assert res["selected_ewma_lambda"] == 0.94
    assert len(res["grid_evaluations"]) > 0


def test_freeze_hyperparameters_artifact(synthetic_test_config, tmp_path):
    """Verify freeze_hyperparameters creates valid JSON artifact with hashes and metadata."""
    target_file = tmp_path / "frozen.json"
    dummy_params = {"selected_window_length": 250, "selected_ewma_lambda": 0.94}

    dest = freeze_hyperparameters(dummy_params, synthetic_test_config, destination_path=target_file)
    assert dest.exists()

    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "FROZEN"
    assert "frozen_at_utc" in data
    assert "config_sha256" in data
    assert data["selected_hyperparameters"] == dummy_params
    assert "disclaimer" in data
