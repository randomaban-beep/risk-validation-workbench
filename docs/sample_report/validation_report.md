# Market Risk Model Validation Report


> **DATA SOURCE: yfinance**


> **Regulatory & Production Disclaimer**:
> Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.

---

## 1. Executive Summary & Algorithmic Findings


- DATA STATUS: Validation executed on real market data (yfinance) covering 12 assets.

- Hyperparameter Grid Search selected a rolling window of 1000 trading days and an EWMA decay factor of 0.97.

- Basel Traffic Light (99% VaR): Models classified in the Green zone (<= 4 exceptions): `parametric_t`.

- Basel Traffic Light (99% VaR): Models in the Yellow zone (supervisory review): `historical_simulation`, `parametric_normal`, `ewma`, `garch_fhs`, `monte_carlo`.

- Kupiec Unconditional Coverage rejection: Model `ewma` failed at alpha=0.05 (p=0.0019) with 28 realized exceptions versus 14.7 expected.

- Christoffersen Independence rejection: Model `historical_simulation` exhibited significant exception clustering (p=0.0041 < 0.05), violating serial independence.

- Christoffersen Independence rejection: Model `parametric_normal` exhibited significant exception clustering (p=0.0058 < 0.05), violating serial independence.

- Christoffersen Independence rejection: Model `monte_carlo` exhibited significant exception clustering (p=0.0058 < 0.05), violating serial independence.

- McNeil-Frey ES Exceedance test rejection: Model `parametric_normal` rejected H0 (p=0.0080 < 0.05), indicating realized tail losses significantly exceeded the Expected Shortfall forecasts.

- McNeil-Frey ES Exceedance test rejection: Model `ewma` rejected H0 (p=0.0026 < 0.05), indicating realized tail losses significantly exceeded the Expected Shortfall forecasts.

- McNeil-Frey ES Exceedance test rejection: Model `monte_carlo` rejected H0 (p=0.0095 < 0.05), indicating realized tail losses significantly exceeded the Expected Shortfall forecasts.

- LIMITATION — McNeil-Frey ES Test Power: With approximately 1466 test-period observations, the expected number of 99% VaR exceedances is approximately 14.7. The McNeil-Frey test has limited statistical power with fewer than ~30 exceedances. Non-rejection should not be interpreted as evidence of good ES calibration.


---

## 2. Indicative Universe & Data Scope

- **Portfolio Name**: `equal_weight`
- **Universe Assets (12)**: `SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, HYG, GLD, DBC`
- **Evaluation Period**: `2019-01-01` to `Latest`

---

## 3. Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | Indep p-val | CC p-val | Basel Zone (worst 250d) | Basel Zone (latest 250d) | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

| **historical_simulation** | 1466 | 13 | 0.89% | 14.7 | 0.6568 | 0.0041 | 0.0148 | **Yellow** | Green | 0.9827 | 0.0186 | 0.0323 |

| **parametric_normal** | 1466 | 14 | 0.95% | 14.7 | 0.8614 | 0.0058 | 0.0219 | **Yellow** | Green | 0.0080 | 0.0173 | 0.0199 |

| **parametric_t** | 1466 | 11 | 0.75% | 14.7 | 0.3149 | 0.0688 | 0.1152 | **Green** | Green | 0.9408 | 0.0201 | 0.0307 |

| **ewma** | 1466 | 28 | 1.91% | 14.7 | 0.0019 | 0.1135 | 0.0023 | **Yellow** | Yellow | 0.0026 | 0.0151 | 0.0173 |

| **garch_fhs** | 1466 | 13 | 0.89% | 14.7 | 0.6568 | 0.1019 | 0.2378 | **Yellow** | Green | 0.3295 | 0.0186 | 0.0232 |

| **monte_carlo** | 1466 | 14 | 0.95% | 14.7 | 0.8614 | 0.0058 | 0.0219 | **Yellow** | Green | 0.0095 | 0.0173 | 0.0199 |


---

## 4. Hyperparameter Selection & Validation Tuning

- **Selected Estimation Window**: `1000` trading days
- **Selected EWMA Lambda**: `0.97`
- **Selection Criterion**: `min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)` (Best Score: `2.0225`)

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
- **ES Test Low Power**: The McNeil-Frey exceedance-residual test has low statistical power when the number of tail exceedances is small (typically < 30). At 1% VaR over ~1,800 test-period days, expected exceedances ≈ 18, which is near the lower bound for reliable inference. Non-rejection should not be interpreted as evidence of good ES calibration.
- **Basel Traffic Light — Rolling Windows**: The standard Basel regime evaluates exceptions over rolling 250-day windows. The "worst zone" column reflects the most adverse window observed during the test period. The "latest zone" reflects the most recent 250 days. These are labelled "traffic-light classification, for illustration".

---

## 7. Reproducibility & Provenance Appendix

| Parameter | Value |
|---|---|
| Report Generated UTC | `2026-10-24T22:21:55.654320+00:00` |
| Config SHA256 | `5e37ba00c7525d16844ad952bfecd2906989d6a110f7c08f09a73c8ae4085d36` |
| Test Period Evaluations Count | `1` |
| Global Seed | `42` |
| Monte Carlo Seed | `1337` |
| Synthetic Seed | `777` |

---
*Market Risk Model Validation Workbench — Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.*