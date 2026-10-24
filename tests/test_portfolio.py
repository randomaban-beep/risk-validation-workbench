"""Unit tests for portfolio modeling, returns aggregation, and period splitting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from rvw.config import load_config
from rvw.exceptions import ConfigurationError, HyperparameterUnfrozenError
from rvw.portfolio.periods import PeriodSplit
from rvw.portfolio.portfolio import Portfolio, create_preset_portfolio, portfolio_returns


def test_portfolio_weight_validation():
    """Verify weight validation and normalisation."""
    p = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="60_40")
    assert np.isclose(p.weights.sum(), 1.0)
    assert p.tickers == ["SPY", "TLT"]

    # Invalid sum must raise ConfigurationError
    with pytest.raises(ConfigurationError):
        Portfolio(weights=pd.Series({"SPY": 0.5, "TLT": 0.2}), name="invalid")


def test_preset_portfolios(default_config):
    """Verify loading standard preset portfolios from config."""
    p_eq = create_preset_portfolio("equal_weight", default_config.portfolios)
    assert len(p_eq.tickers) == 12
    assert np.isclose(p_eq.weights.sum(), 1.0)

    p_6040 = create_preset_portfolio("sixty_forty", default_config.portfolios)
    assert np.isclose(p_6040.weights["SPY"], 0.60)
    assert np.isclose(p_6040.weights["IEF"], 0.40)


def test_portfolio_returns_calculation():
    """Verify return aggregation formula R_p = sum(w_i * R_i)."""
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    ret_df = pd.DataFrame(
        {
            "SPY": [0.01, -0.02],
            "TLT": [0.005, 0.01],
        },
        index=dates,
    )
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="test_p")
    p_ret = portfolio_returns(ret_df, port)

    # Day 0: 0.6 * 0.01 + 0.4 * 0.005 = 0.006 + 0.002 = 0.008
    # Day 1: 0.6 * (-0.02) + 0.4 * 0.01 = -0.012 + 0.004 = -0.008
    assert np.isclose(p_ret.iloc[0], 0.008)
    assert np.isclose(p_ret.iloc[1], -0.008)


def test_portfolio_missing_asset_raises():
    """Verify error when returns dataframe lacks an asset needed by portfolio."""
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    ret_df = pd.DataFrame({"SPY": [0.01, -0.02]}, index=dates)
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="missing")

    with pytest.raises(ConfigurationError):
        portfolio_returns(ret_df, port)


def test_period_split_isolation(sample_returns, default_config):
    """Verify period slicing boundaries."""
    ps = PeriodSplit(sample_returns, default_config)

    train = ps.get_training()
    val = ps.get_validation()

    assert str(train.index.min().date()) >= "2007-06-01"
    assert str(train.index.max().date()) <= "2014-12-31"

    assert str(val.index.min().date()) >= "2015-01-01"
    assert str(val.index.max().date()) <= "2018-12-31"


def test_period_split_freeze_gate(sample_returns, default_config, tmp_path):
    """Verify that test period access refuses unless hyperparameter freeze is satisfied."""
    ps = PeriodSplit(sample_returns, default_config)

    # 1. Config freeze flag is False -> must raise
    assert default_config.freeze_hyperparameters is False
    with pytest.raises(HyperparameterUnfrozenError):
        ps.get_test()

    # 2. Config freeze flag is True but file missing -> must raise
    dummy_frozen_file = tmp_path / "frozen.json"
    default_config.freeze_hyperparameters = True
    default_config.frozen_params_file = str(dummy_frozen_file)

    with pytest.raises(HyperparameterUnfrozenError):
        ps.get_test()

    # 3. Both flag is True and file exists -> succeeds and audits count
    with open(dummy_frozen_file, "w", encoding="utf-8") as f:
        json.dump({"frozen_at": "2026-10-24"}, f)

    test_data = ps.get_test()
    assert len(test_data) > 0
    assert str(test_data.index.min().date()) >= "2019-01-01"
    assert ps.test_evaluation_count == 1

    # Second evaluation increments count
    _ = ps.get_test(caller_context="second_eval")
    assert ps.test_evaluation_count == 2
    assert len(ps.audit_log) == 2
