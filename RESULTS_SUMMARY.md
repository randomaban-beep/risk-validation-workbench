# Market Risk Model Validation Benchmark: Headline Results

**DATA SOURCE: REAL MARKET DATA (yfinance)**
*Run Timestamp*: 2026-10-24T22:21:41.253397+00:00
*Portfolio*: `equal_weight` (12 assets)
*Test Evaluations Count*: 1

> **Disclaimer**: Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.

---

## Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | CC p-val | Basel Zone (worst 250d) | Basel Zone (latest 250d) | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **historical_simulation** | 1466 | 13 | 0.89% | 14.66 | 0.6568 | 0.0148 | **Yellow** | Green | 0.9827 | 0.0186 | 0.0323 |
| **parametric_normal** | 1466 | 14 | 0.95% | 14.66 | 0.8614 | 0.0219 | **Yellow** | Green | 0.0080 | 0.0173 | 0.0199 |
| **parametric_t** | 1466 | 11 | 0.75% | 14.66 | 0.3149 | 0.1152 | **Green** | Green | 0.9408 | 0.0201 | 0.0307 |
| **ewma** | 1466 | 28 | 1.91% | 14.66 | 0.0019 | 0.0023 | **Yellow** | Yellow | 0.0026 | 0.0151 | 0.0173 |
| **garch_fhs** | 1466 | 13 | 0.89% | 14.66 | 0.6568 | 0.2378 | **Yellow** | Green | 0.3295 | 0.0186 | 0.0232 |
| **monte_carlo** | 1466 | 14 | 0.95% | 14.66 | 0.8614 | 0.0219 | **Yellow** | Green | 0.0095 | 0.0173 | 0.0199 |

---

## Hyperparameter Selection (Validation Period Grid Search)
- **Selected Rolling Window**: `1000` trading days
- **Selected EWMA Lambda**: `0.97`
- **Objective Criterion**: `min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)` (Best score: `2.0225`)
