"""Unit tests for Euler risk attribution, what-if rebalancing, and stress testing."""

import numpy as np
import pandas as pd
import pytest

from rvw.attribution.euler import compute_euler_parametric_risk, compute_historical_tail_attribution
from rvw.attribution.what_if import compute_what_if_rebalance
from rvw.portfolio.portfolio import Portfolio
from rvw.stress.covariance_stress import stress_covariance_matrix
from rvw.stress.historical_scenarios import replay_historical_scenario
from rvw.stress.hypothetical import evaluate_hypothetical_scenario


def test_euler_component_var_and_es_exact_summation(sample_returns):
    """Verify Euler decomposition identity: sum(CVaR_i) == Total VaR and sum(CES_i) == Total ES."""
    port = Portfolio(weights=pd.Series({"SPY": 0.35, "QQQ": 0.25, "TLT": 0.25, "GLD": 0.15}), name="test_p")
    window = sample_returns.iloc[:500]

    res = compute_euler_parametric_risk(window, port, alpha=0.99)

    # Absolute sum errors must be virtually zero
    assert res.euler_sum_error_var < 1e-9
    assert res.euler_sum_error_es < 1e-9
    assert np.isclose(sum(res.component_var.values()), res.total_var, atol=1e-8)
    assert np.isclose(sum(res.component_es.values()), res.total_es, atol=1e-8)
    assert np.isclose(sum(res.pct_var.values()), 100.0, atol=1e-3)
    assert np.isclose(sum(res.pct_es.values()), 100.0, atol=1e-3)


def test_historical_tail_attribution_summation(sample_returns):
    """Verify historical tail contributions sum exactly to empirical Expected Shortfall."""
    port = Portfolio(weights=pd.Series({"SPY": 0.4, "TLT": 0.3, "GLD": 0.3}), name="hist_p")
    window = sample_returns.iloc[:400]

    res = compute_historical_tail_attribution(window, port, alpha=0.99)

    assert res["euler_sum_error"] < 1e-9
    assert np.isclose(sum(res["asset_tail_contributions"].values()), res["historical_es"], atol=1e-8)


def test_what_if_rebalancing_risk_monotonicity(sample_returns):
    """Verify increasing equity weight from 50% to 90% increases portfolio VaR and volatility."""
    window = sample_returns.iloc[:400]
    p_curr = Portfolio(weights=pd.Series({"SPY": 0.5, "TLT": 0.5}), name="balanced")
    p_risk = Portfolio(weights=pd.Series({"SPY": 0.9, "TLT": 0.1}), name="risky")

    res = compute_what_if_rebalance(window, p_curr, p_risk, alpha=0.99)

    assert res.target_var > res.current_var
    assert res.delta_var > 0.0
    assert res.delta_vol > 0.0
    assert res.pct_delta_var > 0.0


def test_historical_scenario_replay(sample_returns):
    """Verify historical scenario replay computes cumulative returns and drawdown."""
    port = Portfolio(weights=pd.Series({"SPY": 0.6, "TLT": 0.4}), name="port")
    s_date = str(sample_returns.index[50].date())
    e_date = str(sample_returns.index[70].date())

    res = replay_historical_scenario(sample_returns, port, "test_scen", "Test Scenario", s_date, e_date)

    assert res.trading_days == 21
    assert not np.isnan(res.cumulative_return)
    assert res.max_drawdown >= 0.0
    assert "SPY" in res.asset_cumulative_returns


def test_hypothetical_duration_approximation():
    """Verify duration approximation delta_P / P = - D * delta_y:

    100 bps (+0.01) yield rise with Duration = 17 -> -17% return shock.
    """
    port = Portfolio(weights=pd.Series({"TLT": 1.0}), name="bonds")
    res = evaluate_hypothetical_scenario(
        portfolio=port,
        scenario_id="rates_up",
        scenario_name="Rates +100bps",
        asset_shocks={},
        yield_shocks_bps={"TLT": 100},
        duration_assumptions={"TLT": 17.0},
    )

    assert np.isclose(res.asset_shocks["TLT"], -0.17)
    assert np.isclose(res.portfolio_return, -0.17)
    assert np.isclose(res.portfolio_loss, 0.17)


def test_covariance_stress_increases_var(sample_returns):
    """Verify covariance stressing (vol x1.5, corr +0.3) increases VaR and ES."""
    port = Portfolio(weights=pd.Series({"SPY": 0.5, "TLT": 0.5}), name="port")
    window = sample_returns.iloc[:300]

    res = stress_covariance_matrix(window, port, vol_scale_factor=1.5, correlation_shift=0.3)

    assert res.stressed_vol > res.baseline_vol
    assert res.stressed_var_99 > res.baseline_var_99
    assert res.stressed_es_99 > res.baseline_es_99
    assert res.vol_increase_pct > 0.0
