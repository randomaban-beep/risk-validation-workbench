# Market Risk Model Validation Benchmark: Headline Results

**DATA SOURCE: SYNTHETIC**
*Run Timestamp*: 2026-10-24T18:16:26.684427+00:00
*Portfolio*: `equal_weight` (12 assets)
*Test Evaluations Count*: 1

> **Disclaimer**: Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.

---

## Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | CC p-val | Basel Zone | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|
| **historical_simulation** | 1455 | 12 | 0.82% | 14.55 | 0.4884 | 0.7118 | **Green** | 0.3279 | 0.0178 | 0.0229 |
| **parametric_normal** | 1455 | 14 | 0.96% | 14.55 | 0.8841 | 0.8635 | **Green** | 0.0663 | 0.0177 | 0.0203 |
| **parametric_t** | 1455 | 7 | 0.48% | 14.55 | 0.0269 | 0.0836 | **Green** | 0.7914 | 0.0212 | 0.0306 |
| **ewma** | 1455 | 14 | 0.96% | 14.55 | 0.8841 | 0.0223 | **Green** | 0.0584 | 0.0172 | 0.0197 |
| **garch_fhs** | 1455 | 13 | 0.89% | 14.55 | 0.6775 | 0.8156 | **Green** | 0.1912 | 0.0179 | 0.0219 |
| **monte_carlo** | 1455 | 13 | 0.89% | 14.55 | 0.6775 | 0.8156 | **Green** | 0.0650 | 0.0178 | 0.0204 |

---

## Hyperparameter Selection (Validation Period Grid Search)
- **Selected Rolling Window**: `500` trading days
- **Selected EWMA Lambda**: `0.94`
- **Objective Criterion**: `min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)` (Best score: `3.2203`)
