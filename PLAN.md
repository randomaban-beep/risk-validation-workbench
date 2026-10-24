# Implementation Plan: Market Risk Model Validation Workbench

## 1. Executive Summary & Architecture Overview
The **Market Risk Model Validation Workbench (RVW)** is a modular, production-grade Python framework for market risk model benchmarking, walk-forward out-of-sample forecasting, regulatory-style backtesting (Basel traffic-light, Kupiec, Christoffersen, McNeil-Frey), scenario stress testing, and Euler risk attribution.

The architecture strictly separates:
- **Data Ingestion & Integrity Validation**: Schema validation, price continuity, stale price detection, split/dividend sanity checks, calendar gaps, and structured data-quality audit trail with parquet caching and JSON sidecars.
- **Portfolio & Returns Management**: Portfolio weight validation, portfolio aggregation, and strict period splitting (Training/Warm-up: 2007-06-01 to 2014-12-31, Validation: 2015-01-01 to 2018-12-31, Test: 2019-01-01 onwards) with tamper-resistant hyperparameter freeze enforcement.
- **Risk Calculation Engine**: Six standardized models conforming to a unified functional interface `forecast(returns_window, portfolio, alpha_levels, seed) -> RiskForecast`:
  1. Historical Simulation (empirical quantile, tail-mean ES, configurable interpolation)
  2. Parametric Normal (variance-covariance analytical VaR & ES)
  3. Parametric Student-t (maximum likelihood / method-of-moments fit, analytical VaR & ES)
  4. EWMA Covariance (RiskMetrics-style recursive covariance matrix with normal distribution)
  5. Filtered Historical Simulation (FHS) via GARCH(1,1) Student-t innovations (`arch`)
  6. Monte Carlo Simulation (multivariate normal / Student-t with Cholesky factorization, deterministic seed control)
- **Walk-Forward Engine**: Strict out-of-sample rolling execution (`t` forecast uses strictly $\le t-1$). No look-ahead leakage, refit frequency optimization, structured fit failure logging (no silent fallbacks), and fallback tracking.
- **Statistical Backtesting Suite**:
  - Kupiec Proportion of Failures (Unconditional Coverage LR test)
  - Christoffersen Independence (Markov 1st order transition matrix LR test)
  - Christoffersen Conditional Coverage (joint LR test, $\chi^2$ 2 d.f.)
  - Basel Traffic Light classification (rolling 250-day window 99% VaR)
  - McNeil-Frey Exceedance Residual test for Expected Shortfall
- **Stress Testing & Risk Attribution**:
  - Historical replay scenarios (GFC, COVID-19, 2022 Rates Shock, Aug 2015 flash selloff)
  - Hypothetical multi-asset shocks with duration approximation
  - Correlation & volatility stress shifts
  - Euler component VaR/ES decomposition (verifying $\sum \text{Component} = \text{Total}$) & marginal risk
  - Historical tail risk attribution & what-if rebalancing impact
- **Artifact-Driven Reporting**:
  - Jinja2-rendered interactive HTML report and Markdown report built strictly from saved result artifacts in `results/`.
  - Machine-readable `results/summary.json` and human-readable `RESULTS_SUMMARY.md`.
  - Rule-based findings generator that deduces validation conclusions directly from computed statistics.
- **Streamlit Interactive UI**: Multi-page workbench covering Data Quality, Portfolio Builder, Model Benchmarking, Backtesting & Traffic Light Explorer, Stress Testing, Euler Attribution, and Report Viewer.

---

## 2. Directory Layout
```
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── PLAN.md
├── PROGRESS.md
├── DECISIONS.md
├── RESULTS_SUMMARY.md
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── default.yaml
│   └── synthetic_test.yaml
├── src/
│   └── rvw/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── exceptions.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py
│       │   ├── cache.py
│       │   ├── validator.py
│       │   └── synthetic.py
│       ├── portfolio/
│       │   ├── __init__.py
│       │   ├── portfolio.py
│       │   └── periods.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── historical.py
│       │   ├── parametric_normal.py
│       │   ├── parametric_t.py
│       │   ├── ewma.py
│       │   ├── garch_fhs.py
│       │   └── monte_carlo.py
│       ├── walkforward/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── backtests/
│       │   ├── __init__.py
│       │   ├── kupiec.py
│       │   ├── christoffersen.py
│       │   ├── basel.py
│       │   ├── mcneil_frey.py
│       │   └── suite.py
│       ├── stress/
│       │   ├── __init__.py
│       │   ├── historical_scenarios.py
│       │   ├── hypothetical.py
│       │   └── covariance_stress.py
│       ├── attribution/
│       │   ├── __init__.py
│       │   ├── euler.py
│       │   └── what_if.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   ├── findings.py
│       │   └── templates/
│       │       └── report_template.html
│       └── cli.py
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Data_Quality.py
│       ├── 2_Portfolio_Setup.py
│       ├── 3_Model_Comparison.py
│       ├── 4_Backtest_Results.py
│       ├── 5_Stress_Testing.py
│       ├── 6_Attribution.py
│       └── 7_Report_Export.py
├── tests/
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_portfolio.py
│   ├── test_models_analytical.py
│   ├── test_walkforward.py
│   ├── test_backtests.py
│   ├── test_stress_attribution.py
│   └── test_reporting.py
├── data/
│   ├── sample/
│   │   └── synthetic_prices.parquet
│   └── cache/ (gitignored)
├── results/
├── reports/
├── logs/ (gitignored)
└── docs/
    ├── REPO_METADATA.md
    ├── images/
    └── sample_report/
```

---

## 3. Milestones & Delivery Roadmap

### Milestone 0: Repository Inception & CI Skeleton
- **Deliverables**: `.gitignore`, `LICENSE` (MIT), `pyproject.toml`, `requirements.txt`, `PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, minimal `README.md`, GitHub Actions workflow `.github/workflows/ci.yml`, and `tests/test_smoke.py`.
- **Validation**: `git init`, set remote, initial commit, local `pytest -q` passing, push to GitHub.

### Milestone 1: Scaffold, Config, Structured Logging, Data Pipeline & Integrity Checks
- **Deliverables**: `rvw/config.py` (Pydantic / dataclass / yaml parsing with default fallback), `rvw/logging_utils.py` (dual console + JSONL file logging), `rvw/data/` (Yahoo Finance fetcher with timeout/retry, parquet caching with JSON metadata sidecar, synthetic data generator fallback labelled "SYNTHETIC", and data validator checking duplicates, monotonic dates, calendar gaps, non-positive prices, stale prices, and extreme jumps).
- **Validation**: Tests in `tests/test_config.py`, `tests/test_data.py`. Verify synthetic dataset generation and committed parquet fixture.

### Milestone 2: Portfolio Framework, Returns Engine & Time Period Splitter
- **Deliverables**: `rvw/portfolio/portfolio.py` (Portfolio dataclass, weight sum verification, presets: Equal Weight, 60/40, Risk-Tilted, Equity-Heavy), returns computation (simple arithmetic returns), `rvw/portfolio/periods.py` (`PeriodSplit` enforcing Training 2007-2014, Validation 2015-2018, Test 2019-present, freeze gate requiring frozen hyperparameter artifact before test execution).
- **Validation**: Tests in `tests/test_portfolio.py` verifying weight normalization, portfolio returns arithmetic, period slicing, and freeze locking.

### Milestone 3: Comprehensive Risk Calculation Engine (6 Core Models)
- **Deliverables**:
  1. Historical Simulation: quantile interpolation doc, tail-mean ES.
  2. Parametric Normal: standard normal quantile $z_\alpha$, $\text{ES}_\alpha = \mu + \sigma \frac{\phi(z_\alpha)}{1-\alpha}$.
  3. Parametric Student-$t$: MLE/moments fitting for df $\nu$, analytical Student-$t$ ES.
  4. EWMA Covariance: recursive covariance estimation ($\lambda=0.94, 0.97$), analytical Normal VaR/ES.
  5. Filtered Historical Simulation (FHS): univariate/portfolio GARCH(1,1)-$t$ innovations scaling.
  6. Monte Carlo: Cholesky decomposition, multivariate Normal & multivariate $t$, explicit RNG seeding.
- **Validation**: `tests/test_models_analytical.py` testing against exact known formulas (e.g., $z_{0.99}=2.3263479$, $\text{ES}_{0.99}=2.6652142$, $z_{0.95}=1.6448536$, Euler exact summation, linear homogeneity $\text{VaR}(2w) = 2\text{VaR}(w)$, and Monte Carlo convergence).

### Milestone 4: Strict Walk-Forward Simulation Engine
- **Deliverables**: `rvw/walkforward/engine.py` supporting rolling and expanding windows, `refit_every` parameter, strict $t-1$ cutoff, failure logging without silent masking, fallback counting.
- **Validation**: `tests/test_walkforward.py` verifying no-look-ahead data modification invariance (altering data at $t+1$ does not alter forecast at $t$), failure handling and metadata propagation.

### Milestone 5: Regulatory & Statistical Backtesting Suite
- **Deliverables**:
  1. Kupiec POF Likelihood Ratio test (handles $x=0, x=N$).
  2. Christoffersen Independence test (handles empty state transitions).
  3. Christoffersen Conditional Coverage test ($\text{LR}_{cc} = \text{LR}_{uc} + \text{LR}_{ind}$, 2 d.f.).
  4. Basel 250-day rolling Traffic Light classification (Green: 0-4, Yellow: 5-9, Red: 10+).
  5. McNeil-Frey ES Exceedance Residual test (standardized residuals of losses exceeding VaR, bootstrap/t-test $H_0: \mathbb{E}[R]=0$).
  6. Aggregate benchmark comparison matrix generator.
- **Validation**: `tests/test_backtests.py` testing exact theoretical values (e.g. Kupiec $N=250, x=5 \implies \text{LR}=1.957, p=0.162$), transition matrix edge cases, and traffic light boundary conditions.

### Milestone 6: Hyperparameter Tuning, Selection & Validation Freeze
- **Deliverables**: Validation period (2015-2018) benchmarking over window sizes ($250, 500, 1000$), EWMA $\lambda$ ($0.94, 0.97$), Monte Carlo paths. Hyperparameter selection criterion (minimizing conditional coverage distance and loss tail error). Export `frozen_hyperparameters.json`. Test period run (2019-present) unlocked and evaluated exactly once.
- **Validation**: Automated script / CLI pipeline `rvw run --freeze` verifying freeze gate enforcement and logging audit count.

### Milestone 7: Stress Testing & Euler Risk Attribution
- **Deliverables**:
  - `rvw/stress/`: Historical scenario replays (GFC, COVID-19, 2022 Rates, Aug 2015), hypothetical shocks with modified duration bond repricing, and stressed covariance matrix simulation.
  - `rvw/attribution/`: Parametric Euler component VaR/ES ($\text{CVaR}_i = w_i \frac{\partial \text{VaR}}{\partial w_i}$ summing to $100\%$), historical tail loss attribution, and what-if portfolio rebalancing delta.
- **Validation**: `tests/test_stress_attribution.py` verifying Euler sum identity $\sum \text{CVaR}_i = \text{Total VaR}$, duration sensitivity approximations, and scenario loss monotonicity.

### Milestone 8: Artifact-Driven Report Generation & Automated Findings
- **Deliverables**: `rvw/reporting/` reading only saved parquet/json artifacts from `results/`. Jinja2 templates for HTML and Markdown generation. Algorithmic findings generator extracting bulleted qualitative evaluations from backtest statistics. Export `RESULTS_SUMMARY.md` and `results/summary.json`.
- **Validation**: `tests/test_reporting.py` verifying no fabricated numbers exist outside artifacts and mandatory regulatory disclaimer is present.

### Milestone 9: Streamlit Validation Dashboard
- **Deliverables**: `app/streamlit_app.py` and dedicated sub-pages (`pages/1_*.py` through `pages/7_*.py`) with interactive Plotly visualizers (time series VaR envelopes, hit charts, Basel traffic light status, stress waterfall, Euler attribution bar charts). Pure presentation layer consuming core library functions, zero financial logic embedded in UI.
- **Validation**: Smoke test app imports and page rendering verification.

### Milestone 10: Final Polish, Documentation, Packaging & Release
- **Deliverables**: Comprehensive `README.md` with CI badge, quickstart commands, methodology guide, formula definitions, references, and disclaimer; `docs/REPO_METADATA.md`; sample reports; tag `v0.1.0`; final test verification.
- **Validation**: `pytest -q` running 100% green; git push to origin `main`.
