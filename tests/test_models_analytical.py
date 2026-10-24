"""Analytical reference checks and numerical precision tests for all risk models."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rvw.models.ewma import EWMAModel, compute_ewma_covariance
from rvw.models.garch_fhs import GARCHFilteredHistoricalModel
from rvw.models.historical import HistoricalSimulationModel
from rvw.models.monte_carlo import MonteCarloRiskModel
from rvw.models.parametric_normal import ParametricNormalModel
from rvw.models.parametric_t import ParametricStudentTModel
from rvw.portfolio.portfolio import Portfolio


def test_standard_normal_analytical_reference():
    """Verify known closed-form values for standard normal quantiles and ES.

    Reference values for N(0, 1):
    - z_0.99 = 2.32634787
    - z_0.95 = 1.64485363
    - ES_0.99 = phi(z_0.99) / 0.01 = 2.6652142
    - ES_0.975 = phi(z_0.975) / 0.025 = 2.3378028
    """
    z_99 = stats.norm.ppf(0.99)
    z_95 = stats.norm.ppf(0.95)
    z_975 = stats.norm.ppf(0.975)

    assert np.isclose(z_99, 2.32634787, atol=1e-5)
    assert np.isclose(z_95, 1.64485363, atol=1e-5)

    es_99 = stats.norm.pdf(z_99) / 0.01
    es_975 = stats.norm.pdf(z_975) / 0.025

    assert np.isclose(es_99, 2.6652142, atol=1e-5)
    assert np.isclose(es_975, 2.3378028, atol=1e-5)


def test_parametric_normal_model_forecast():
    """Verify ParametricNormalModel matches analytical formulas."""
    rng = np.random.default_rng(42)
    T = 200
    mock_returns = pd.DataFrame({"SPY": rng.normal(loc=0.001, scale=0.015, size=T)})
    portfolio = Portfolio(weights=pd.Series({"SPY": 1.0}), name="single_asset")

    model = ParametricNormalModel(params={"assume_zero_mean": False})
    forecast = model.forecast(mock_returns, portfolio, var_alphas=[0.95, 0.99], es_alphas=[0.975, 0.99])

    sample_mu = float(mock_returns["SPY"].mean())
    sample_sigma = float(mock_returns["SPY"].std(ddof=1))

    expected_var_99 = -sample_mu + sample_sigma * stats.norm.ppf(0.99)
    expected_es_99 = -sample_mu + sample_sigma * (stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)

    assert np.isclose(forecast.get_var(0.99), expected_var_99, atol=1e-7)
    assert np.isclose(forecast.get_es(0.99), expected_es_99, atol=1e-7)
    assert forecast.get_es(0.99) >= forecast.get_var(0.99)


def test_student_t_analytical_es_formula():
    """Verify analytical Student-t ES matches numerical integration."""
    nu = 5.0
    alpha = 0.99
    q = stats.t.ppf(alpha, df=nu)

    # Closed-form formula
    pdf_q = stats.t.pdf(q, df=nu)
    closed_form_es = (pdf_q / (1.0 - alpha)) * ((nu + q**2) / (nu - 1.0))

    # Numerical integration of tail expectation: \int_q^\infty x f(x) dx / (1 - alpha)
    from scipy.integrate import quad
    integral, _ = quad(lambda x: x * stats.t.pdf(x, df=nu), q, np.inf)
    numerical_es = integral / (1.0 - alpha)

    assert np.isclose(closed_form_es, numerical_es, atol=1e-5)


def test_linear_homogeneity_scaling():
    """Verify linear scaling property VaR(c * X) = c * VaR(X) for zero-mean models."""
    rng = np.random.default_rng(101)
    T = 300
    base_returns = rng.normal(loc=0.0, scale=0.01, size=T)

    df_base = pd.DataFrame({"A": base_returns})
    df_double = pd.DataFrame({"A": 2.0 * base_returns})
    port = Portfolio(weights=pd.Series({"A": 1.0}), name="port")

    norm_model = ParametricNormalModel(params={"assume_zero_mean": True})
    fc_base = norm_model.forecast(df_base, port, var_alphas=[0.99], es_alphas=[0.99])
    fc_double = norm_model.forecast(df_double, port, var_alphas=[0.99], es_alphas=[0.99])

    assert np.isclose(fc_double.get_var(0.99), 2.0 * fc_base.get_var(0.99), atol=1e-6)
    assert np.isclose(fc_double.get_es(0.99), 2.0 * fc_base.get_es(0.99), atol=1e-6)


def test_historical_simulation_quantile_and_tail_mean():
    """Verify HS VaR matches numpy quantile and ES is the tail average."""
    # Create distinct sorted losses
    losses = np.linspace(0.01, 0.10, 100)
    # returns = -losses
    df_ret = pd.DataFrame({"SPY": -losses})
    port = Portfolio(weights=pd.Series({"SPY": 1.0}), name="hs_test")

    model = HistoricalSimulationModel(params={"interpolation": "linear"})
    fc = model.forecast(df_ret, port, var_alphas=[0.95], es_alphas=[0.95])

    expected_var = np.percentile(losses, 95.0, method="linear")
    tail = losses[losses >= expected_var]
    expected_es = np.mean(tail)

    assert np.isclose(fc.get_var(0.95), expected_var)
    assert np.isclose(fc.get_es(0.95), expected_es)
    assert fc.get_es(0.95) >= fc.get_var(0.95)


def test_ewma_recursion_matches_manual_calculation():
    """Verify EWMA recursion equals explicit step-by-step calculation on small dataset."""
    r = np.array([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.02]])
    lam = 0.94
    initial_cov = np.cov(r, rowvar=False)

    # Step 0
    cov0 = initial_cov.copy()
    # Step 1
    r0 = r[0:1, :]
    cov1 = lam * cov0 + (1 - lam) * (r0.T @ r0)
    # Step 2
    r1 = r[1:2, :]
    cov2 = lam * cov1 + (1 - lam) * (r1.T @ r1)
    # Step 3
    r2 = r[2:3, :]
    cov3 = lam * cov2 + (1 - lam) * (r2.T @ r2)

    computed = compute_ewma_covariance(r, lambda_param=lam, initial_cov=initial_cov)
    np.testing.assert_allclose(computed, cov3, atol=1e-10)


def test_monte_carlo_determinism_and_convergence():
    """Verify Monte Carlo is bit-identical with same seed, different with different seeds,

    and converges to analytical normal VaR as N increases.
    """
    rng = np.random.default_rng(999)
    T = 250
    returns = pd.DataFrame({"SPY": rng.normal(0.0, 0.02, size=T)})
    port = Portfolio(weights=pd.Series({"SPY": 1.0}), name="mc_test")

    mc_model = MonteCarloRiskModel(params={"num_simulations": 10000, "distribution": "normal"})
    fc1 = mc_model.forecast(returns, port, seed=12345)
    fc2 = mc_model.forecast(returns, port, seed=12345)
    fc3 = mc_model.forecast(returns, port, seed=67890)

    # Bit-identical with same seed
    assert fc1.get_var(0.99) == fc2.get_var(0.99)
    assert fc1.get_es(0.99) == fc2.get_es(0.99)

    # Different seeds produce different realizations
    assert fc1.get_var(0.99) != fc3.get_var(0.99)

    # Convergence to analytical normal for large N
    mc_large = MonteCarloRiskModel(params={"num_simulations": 40000, "distribution": "normal"})
    fc_large = mc_large.forecast(returns, port, seed=42)

    norm_model = ParametricNormalModel(params={"assume_zero_mean": False})
    fc_norm = norm_model.forecast(returns, port)

    # Within statistical standard error of simulation
    assert np.isclose(fc_large.get_var(0.99), fc_norm.get_var(0.99), atol=1e-3)
    assert np.isclose(fc_large.get_es(0.99), fc_norm.get_es(0.99), atol=1e-3)


def test_garch_fhs_model_execution(sample_returns):
    """Verify GARCH FHS runs on sample returns and produces valid forecasts."""
    port = Portfolio(weights=pd.Series({"SPY": 0.5, "TLT": 0.5}), name="port")
    window = sample_returns.iloc[:250]

    model = GARCHFilteredHistoricalModel(params={"p": 1, "q": 1, "resample_size": 2000})
    fc = model.forecast(window, port, seed=42)

    assert fc.model_name == "garch_fhs"
    assert 0.95 in fc.var_forecasts
    assert 0.99 in fc.var_forecasts
    assert fc.get_es(0.99) >= fc.get_var(0.99)
    assert "cond_sigma" in fc.metadata
