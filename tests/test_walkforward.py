"""Unit tests for walk-forward execution, no-look-ahead integrity, and failure handling."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

from rvw.exceptions import ModelFitError
from rvw.models.base import BaseRiskModel, RiskForecast
from rvw.models.historical import HistoricalSimulationModel
from rvw.models.parametric_normal import ParametricNormalModel
from rvw.portfolio.portfolio import Portfolio
from rvw.walkforward.engine import run_walkforward


class FlakyMockModel(BaseRiskModel):
    """Mock model designed to fail on specific dates to test exception handling."""

    def __init__(self, fail_step: int = 5) -> None:
        super().__init__(name="flaky_mock")
        self.call_count = 0
        self.fail_step = fail_step

    def forecast(
        self,
        returns_window: pd.DataFrame,
        portfolio: Portfolio,
        var_alphas: Optional[List[float]] = None,
        es_alphas: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> RiskForecast:
        self.call_count += 1
        if self.call_count == self.fail_step:
            raise ModelFitError(f"Simulated failure on call {self.call_count}")

        return RiskForecast(
            model_name=self.name,
            var_forecasts={0.95: 0.02, 0.99: 0.03},
            es_forecasts={0.975: 0.025, 0.99: 0.035},
        )


def test_no_look_ahead_causality_leakage(sample_returns):
    """Verify modifying future return data at t+5 does not alter forecasts at <= t+5."""
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="port")
    window = 100

    base_sample = sample_returns.iloc[:150].copy()
    model = ParametricNormalModel()

    # Base walkforward
    ff_base = run_walkforward(base_sample, port, model, window_length=window, refit_every=1)

    # Contaminate future day 120 (forecast index 20) with a wild return
    corrupted_sample = base_sample.copy()
    corrupted_sample.iloc[120, :] = 0.50  # 50% jump in future

    ff_corrupted = run_walkforward(corrupted_sample, port, model, window_length=window, refit_every=1)

    # Forecasts for all dates PRIOR TO OR ON day 120 (which only used data up to day 119)
    # MUST be strictly identical!
    dates_before = base_sample.index[window:121]
    pd.testing.assert_frame_equal(
        ff_base.var_forecasts.loc[dates_before],
        ff_corrupted.var_forecasts.loc[dates_before],
    )


def test_failure_handling_and_fallback_tracking(sample_returns):
    """Verify that model fitting errors are captured, logged, and tracked without masking."""
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="port")
    sample_sub = sample_returns.iloc[:80].copy()
    window = 50

    # 1. Fallback enabled
    flaky1 = FlakyMockModel(fail_step=3)
    ff_fallback = run_walkforward(
        sample_sub,
        port,
        flaky1,
        window_length=window,
        refit_every=1,
        fallback_on_failure=True,
    )
    assert len(ff_fallback.failed_dates) == 1
    assert ff_fallback.fallback_count == 1
    # Fallback model provided values instead of NaN
    assert not np.isnan(ff_fallback.var_forecasts.iloc[2, 0])

    # 2. Fallback disabled -> records NaN with failed_dates
    flaky2 = FlakyMockModel(fail_step=3)
    ff_no_fallback = run_walkforward(
        sample_sub,
        port,
        flaky2,
        window_length=window,
        refit_every=1,
        fallback_on_failure=False,
    )
    assert len(ff_no_fallback.failed_dates) == 1
    assert ff_no_fallback.fallback_count == 0
    assert np.isnan(ff_no_fallback.var_forecasts.iloc[2, 0])


def test_refit_frequency_behavior(sample_returns):
    """Verify that refit_every=5 only calls model.forecast every 5 days."""
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="port")
    sample_sub = sample_returns.iloc[:70].copy()
    window = 50  # 20 forecast days
    n_forecast_days = len(sample_sub) - window

    mock = FlakyMockModel(fail_step=999)  # Never fails
    _ = run_walkforward(sample_sub, port, mock, window_length=window, refit_every=5)

    expected_calls = int(np.ceil(n_forecast_days / 5))
    assert mock.call_count == expected_calls
