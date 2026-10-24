# Market Risk Model Validation Benchmark: Headline Results

**DATA SOURCE: REAL MARKET DATA (yfinance)**
*Run Timestamp*: 2026-10-24T22:40:51.059765+00:00
*Portfolio*: `equal_weight` (12 assets)
*Test Evaluations Count*: 1

> **Disclaimer**: Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.

---

## Out-of-Sample Test Period Performance (99% VaR & 99% ES)

| Model Name | Obs | Hits | Hit Rate | Expected | Kupiec p-val | CC p-val | Basel Zone (worst 250d) | Basel Zone (latest 250d) | McNeil-Frey p-val | Mean VaR | Mean ES |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **historical_simulation** | 1964 | 25 | 1.27% | 19.64 | 0.2435 | 0.0005 | **Red** | Green | 0.3175 | 0.0177 | 0.0305 |
| **parametric_normal** | 1964 | 31 | 1.58% | 19.64 | 0.0175 | 0.0003 | **Red** | Green | 0.0008 | 0.0162 | 0.0186 |
| **parametric_t** | 1964 | 22 | 1.12% | 19.64 | 0.5995 | 0.0053 | **Red** | Green | 0.0623 | 0.0193 | 0.0306 |
| **ewma** | 1964 | 43 | 2.19% | 19.64 | 0.0000 | 0.0000 | **Red** | Green | 0.0002 | 0.0158 | 0.0181 |
| **garch_fhs** | 1964 | 25 | 1.27% | 19.64 | 0.2435 | 0.0065 | **Yellow** | Green | 0.1265 | 0.0188 | 0.0235 |
| **monte_carlo** | 1964 | 31 | 1.58% | 19.64 | 0.0175 | 0.0003 | **Red** | Green | 0.0008 | 0.0162 | 0.0186 |

---

## Hyperparameter Selection (Validation Period Grid Search)
- **Selected Rolling Window**: `1000` trading days
- **Selected EWMA Lambda**: `0.97`
- **Objective Criterion**: `min(|actual_rate - 0.01| * 100 + 0.5 * LR_cc)` (Best score: `2.0225`)
