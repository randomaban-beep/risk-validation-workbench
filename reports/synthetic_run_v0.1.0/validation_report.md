# Market Risk Model Validation Report


> **DATA SOURCE: SYNTHETIC**


> **Regulatory & Production Disclaimer**:
> Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.

---

## 1. Executive Summary & Algorithmic Findings


- DATA STATUS: Validation was executed on calibrated SYNTHETIC market data. Statistical outcomes demonstrate code execution and pipeline integrity, not real-world market dynamics.

- Hyperparameter Grid Search selected a rolling window of 500 trading days and an EWMA decay factor of 0.94.

- Basel Traffic Light (99% VaR): Models classified in the Green zone (<= 4 exceptions): `historical_simulation`, `parametric_normal`, `parametric_t`, `ewma`, `garch_fhs`, `monte_carlo`.

- Kupiec Unconditional Coverage rejection: Model `parametric_t` failed at alpha=0.05 (p=0.0269) with 7 realized exceptions versus 14.6 expected.

- Christoffersen Independence rejection: Model `ewma` exhibited significant exception clustering (p=0.0059 < 0.05), violating serial independence.

- McNeil-Frey ES Exceedance: No model showed statistically significant tail underestimation at alpha=0.05.


---

## 2. Indicative Universe & Data Scope

- **Portfolio Name**: `equal_weight`
- **Universe Assets (12)**: `SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, HYG, GLD, DBC`
- **Evaluation Period**: `2019-01-01` to `Latest`

---

## 3. Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | Indep p-val | CC p-val | Basel Zone | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|---|

| **historical_simulation** | 1455 | 12 | 0.82% | 14.6 | 0.4884 | 0.6549 | 0.7118 | **Green** | 0.3279 | 0.0178 | 0.0229 |

| **parametric_normal** | 1455 | 14 | 0.96% | 14.6 | 0.8841 | 0.6018 | 0.8635 | **Green** | 0.0663 | 0.0177 | 0.0203 |

| **parametric_t** | 1455 | 7 | 0.48% | 14.6 | 0.0269 | 0.7947 | 0.0836 | **Green** | 0.7914 | 0.0212 | 0.0306 |

| **ewma** | 1455 | 14 | 0.96% | 14.6 | 0.8841 | 0.0059 | 0.0223 | **Green** | 0.0584 | 0.0172 | 0.0197 |

| **garch_fhs** | 1455 | 13 | 0.89% | 14.6 | 0.6775 | 0.6282 | 0.8156 | **Green** | 0.1912 | 0.0179 | 0.0219 |

| **monte_carlo** | 1455 | 13 | 0.89% | 14.6 | 0.6775 | 0.6282 | 0.8156 | **Green** | 0.0650 | 0.0178 | 0.0204 |


---

## 4. Hyperparameter Selection & Validation Tuning

- **Selected Estimation Window**: `500` trading days
- **Selected EWMA Lambda**: `0.94`
- **Selection Criterion**: `min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)` (Best Score: `3.2203`)

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
| Report Generated UTC | `2026-10-24T18:16:31.469345+00:00` |
| Config SHA256 | `5e37ba00c7525d16844ad952bfecd2906989d6a110f7c08f09a73c8ae4085d36` |
| Test Period Evaluations Count | `1` |
| Global Seed | `42` |
| Monte Carlo Seed | `1337` |
| Synthetic Seed | `777` |

---
*Market Risk Model Validation Workbench — Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.*