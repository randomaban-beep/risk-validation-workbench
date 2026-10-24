# Market Risk Model Validation Report

{% if is_synthetic %}
> **DATA SOURCE: SYNTHETIC**
{% else %}
> **DATA SOURCE: {{ source }}**
{% endif %}

> **Regulatory & Production Disclaimer**:
> {{ disclaimer }}

---

## 1. Executive Summary & Algorithmic Findings

{% for finding in findings %}
- {{ finding }}
{% endfor %}

---

## 2. Indicative Universe & Data Scope

- **Portfolio Name**: `{{ portfolio_name }}`
- **Universe Assets ({{ tickers|length }})**: `{{ tickers|join(", ") }}`
- **Evaluation Period**: `{{ start_date }}` to `{{ end_date }}`

---

## 3. Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | Indep p-val | CC p-val | Basel Zone | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|---|
{% for name, m in test_metrics.items() %}
| **{{ name }}** | {{ m.observations }} | {{ m.exceptions }} | {{ "%.2f"|format(m.exception_rate * 100) }}% | {{ "%.1f"|format(m.expected_exceptions) }} | {{ "%.4f"|format(m.kupiec_p) }} | {{ "%.4f"|format(m.independence_p) }} | {{ "%.4f"|format(m.conditional_coverage_p) }} | **{{ m.basel_zone }}** | {{ "%.4f"|format(m.mcneil_frey_p) }} | {{ "%.4f"|format(m.mean_var) }} | {{ "%.4f"|format(m.mean_es) }} |
{% endfor %}

---

## 4. Hyperparameter Selection & Validation Tuning

- **Selected Estimation Window**: `{{ hyperparameter_selection.selected_window_length }}` trading days
- **Selected EWMA Lambda**: `{{ hyperparameter_selection.selected_ewma_lambda }}`
- **Selection Criterion**: `{{ hyperparameter_selection.selection_metric }}` (Best Score: `{{ hyperparameter_selection.best_score }}`)

---

## 5. Model Methodologies & Statistical Assumptions

1. **Historical Simulation (HS)**: Empirical quantile of rolling historical losses with linear interpolation. Expected Shortfall estimated as empirical conditional tail mean.
2. **Parametric Normal**: Analytical Gaussian variance-covariance model with standard normal quantile and PDF tail expectation.
3. **Parametric Student-t**: Maximum Likelihood Estimation of degrees of freedom nu; analytical Student-t quantile and exact closed-form expected shortfall.
4. **EWMA (RiskMetrics)**: Recursive exponential decay covariance matrix with normal distribution assumption.
5. **Filtered Historical Simulation (FHS)**: GARCH(1,1) with Student-t innovations; standardized residuals resampled and scaled by 1-day ahead conditional volatility.
6. **Monte Carlo Simulation**: Cholesky factorization of empirical covariance matrix with multivariate normal simulation under explicit deterministic RNG seed.

---

## 6. Limitations and Known Weaknesses

- **Linearity & Rebalancing**: Daily portfolio weights are assumed to rebalance without friction, market impact, or transaction costs.
- **Horizon Extrapolation**: 1-day horizon risk metrics cannot be scaled to 10-day regulatory periods via square-root-of-time during volatile regimes.
- **Fat-tail Risk**: Normal variance-covariance models systematically underestimate tail severity during market crises.
- **Asymmetric Correlation**: Joint tail dependence is not captured by linear correlation matrices.

---

## 7. Reproducibility & Provenance Appendix

| Parameter | Value |
|---|---|
| Report Generated UTC | `{{ generated_at_utc }}` |
| Config SHA256 | `{{ config_hash }}` |
| Test Period Evaluations Count | `{{ test_evaluations_count }}` |
| Global Seed | `{{ seeds.global_seed }}` |
| Monte Carlo Seed | `{{ seeds.monte_carlo_seed }}` |
| Synthetic Seed | `{{ seeds.synthetic_seed }}` |

---
*Market Risk Model Validation Workbench — Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.*
