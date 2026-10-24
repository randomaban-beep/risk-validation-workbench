# Architectural & Methodological Decisions

This document records all key technical, architectural, and mathematical decisions made during the design and implementation of the Market Risk Model Validation Workbench.

---

## 1. Sign and Unit Conventions
- **Loss Definition**: Loss is defined as $L_t = -R_{p,t}$, where $R_{p,t}$ is the portfolio simple return on day $t$.
- **VaR & ES Sign**: Value at Risk ($\text{VaR}_\alpha$) and Expected Shortfall ($\text{ES}_\alpha$) are strictly reported as positive numbers representing a fractional loss of portfolio value (e.g. $0.025$ means a $2.5\%$ loss).
- **Horizon**: Exactly 1 trading day ($h = 1$). No square-root-of-time scaling across multiple days is applied unless explicitly part of an illustrative stress scenario.
- **Coverage Levels**: VaR is evaluated at $\alpha \in \{0.95, 0.99\}$ (5% and 1% left tails of return distribution, corresponding to 95% and 99% right tails of loss distribution). ES is evaluated at $\alpha \in \{0.975, 0.99\}$ conforming to Basel III / FRTB standards.
- **Hit Definition**: A VaR exception (hit) occurs on day $t$ if and only if $L_t > \text{VaR}_{\alpha, t|t-1}$.

---

## 2. Dependency Pinning & Environment
- **Python Version**: Python 3.11.9.
- **Core Dependencies**: Exact pinned versions in `requirements.txt` matching the tested runtime environment:
  - `numpy==2.3.5`
  - `pandas==2.3.3`
  - `scipy==1.17.1`
  - `statsmodels==0.15.0`
  - `arch==8.0.0`
  - `streamlit==1.58.0`
  - `plotly==5.22.0`
  - `pytest==9.1.1`
  - `pyyaml==6.0.3`
  - `yfinance==1.7.0`
  - `pyarrow==24.0.0`
  - `jinja2==3.1.6`
  - `click==8.4.2`

---

## 3. Data Integrity & Caching Strategy
- **Indicative Universe**: 12 liquid ETFs representing diverse global asset classes: `SPY` (US Large Cap), `QQQ` (US Tech), `IWM` (US Small Cap), `EFA` (Dev Ex-US), `EEM` (Emerging Markets), `VNQ` (Real Estate), `TLT` (20+ Yr Treasury), `IEF` (7-10 Yr Treasury), `LQD` (IG Corporate Bond), `HYG` (High Yield Bond), `GLD` (Gold), `DBC` (Commodities).
- **Cache Format**: Local Apache Parquet with sidecar `.meta.json` in `data/cache/`. Gitignored.
- **Network Resilience Policy**: If remote download fails, system gracefully falls back to synthetic geometric Brownian motion / t-copula data with explicit `"SYNTHETIC"` metadata tags displayed across all logs, dataframes, and reports.

---

## 4. Model Specifications & Mathematical Formulations
1. **Historical Simulation (HS)**:
   - Rolling empirical quantile of negative returns with linear interpolation: $\text{VaR}_\alpha = Q_\alpha(-R_p)$.
   - $\text{ES}_\alpha = \frac{1}{K} \sum_{i: L_i > \text{VaR}_\alpha} L_i$.
2. **Parametric Normal**:
   - $\text{VaR}_\alpha = -\mu_p + \sigma_p z_\alpha$, where $z_\alpha = \Phi^{-1}(\alpha)$.
   - $\text{ES}_\alpha = -\mu_p + \sigma_p \frac{\phi(z_\alpha)}{1 - \alpha}$.
3. **Parametric Student-t**:
   - Degrees of freedom $\nu$ estimated via MLE on standardized return residuals.
   - Scale $\tilde{\sigma} = \sigma_p \sqrt{\frac{\nu - 2}{\nu}}$.
   - $\text{VaR}_\alpha = -\mu_p + \tilde{\sigma} t_\nu^{-1}(\alpha)$.
   - $\text{ES}_\alpha = -\mu_p + \tilde{\sigma} \frac{f_\nu(t_\nu^{-1}(\alpha))}{1 - \alpha} \left(\frac{\nu + (t_\nu^{-1}(\alpha))^2}{\nu - 1}\right)$.
4. **EWMA (RiskMetrics)**:
   - Recursive covariance matrix: $\Sigma_t = \lambda \Sigma_{t-1} + (1-\lambda) r_{t-1} r_{t-1}^T$.
   - Normal distribution assumption for analytical quantile and expected shortfall.
5. **Filtered Historical Simulation (FHS)**:
   - GARCH(1,1) with Student-t innovations fitted to portfolio return series using `arch`.
   - Scaled standardized residual draws projected with conditional variance $\sigma_{t|t-1}$.
6. **Monte Carlo Simulation**:
   - Multivariate Normal and Student-t generation using Cholesky factor $L$ ($\Sigma = L L^T$).
   - Explicit seed parameter with `numpy.random.default_rng(seed)` ensuring deterministic outputs.

---

## 5. Walk-Forward Simulation & Freeze Protocol
- **Strict Information Cutoff**: Model forecast for day $t$ utilizes historical sample up to and including day $t-1$. Zero look-ahead leakage.
- **Period Splitting**:
  - Training/Warm-up: 2007-06-01 to 2014-12-31
  - Validation: 2015-01-01 to 2018-12-31
  - Test: 2019-01-01 to end
- **Freeze Enforcement**: Evaluating the Test period raises `HyperparameterUnfrozenError` unless `config.freeze_hyperparameters = True` and a signed `frozen_hyperparameters.json` file is present on disk. All evaluations of the test period are logged with evaluation counts and timestamps.

---

## 6. Backtesting Statistics
- **Kupiec POF**: Log-likelihood ratio test of unconditional coverage $\text{LR}_{uc} \sim \chi^2(1)$. Tested at edge cases $x=0$ and $x=N$.
- **Christoffersen Independence**: First-order Markov chain transition probability test $\text{LR}_{ind} \sim \chi^2(1)$.
- **Christoffersen Conditional Coverage**: $\text{LR}_{cc} = \text{LR}_{uc} + \text{LR}_{ind} \sim \chi^2(2)$.
- **Basel Traffic Light**: Binomial cumulative distribution over 250 rolling trading days: Green ($0-4$), Yellow ($5-9$), Red ($10+$).
- **McNeil-Frey ES Test**: One-sided hypothesis test on standardized tail exceedance residuals $R_{t} = \frac{L_t - \text{ES}_t}{\sigma_t}$, testing $H_0: \mathbb{E}[R] = 0$.

---

## 7. Stress Testing & Euler Attribution
- **Euler Allocation**: Homogeneity of degree 1 holds for VaR: $\text{VaR}(w) = \sum_{i=1}^N w_i \frac{\partial \text{VaR}}{\partial w_i}$. Component VaR sums exactly to total VaR.
- **Scenarios**: Historical replays take exact historical realized return vectors; hypothetical scenarios apply asset-level shocks with bond duration adjustments.

---

## 8. Regulatory Disclaimer
Every report, dashboard page, CLI output, and README prominently displays:
> "Educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation."

---

## 9. Real-Data Fetch — Data Quality Log (2026-10-25)
- **Source**: yfinance (bypass-sandbox mode required; sandboxed network blocked DNS)
- **Download timestamp**: 2026-10-25 (UTC)
- **Tickers fetched**: All 12 (SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, HYG, GLD, DBC)
- **Rows**: 4,881 trading days
- **Date range**: 2007-06-01 to 2026-10-23
- **Ticker start-date issues**: None — all 12 tickers start at the configured 2007-06-01
- **Substitutions**: None required
- **NaNs**: Zero across all tickers
- **Non-positive prices**: Zero
- **Extreme returns (|r| > 25%)**: Zero days across all tickers
- **Data quality report outcome**: CLEAN — no issues logged

---

## 10. Basel Traffic Light — Full-Period vs. Rolling Window (2026-10-25)
The `evaluate_backtests()` function in `suite.py` calls `classify_basel_zone(n_exc, window_length=n_obs)` where
`n_obs` is the full test-period observation count (1,964 days), NOT a fixed 250-day window.
This means the zone boundary scales proportionally (`green_cutoff = floor(4 * n_obs/250)`), which is NOT the
standard Basel interpretation (which always uses a fixed 250-day window).

**Fix applied**: The suite now reports the zone from the worst and latest 250-day rolling windows using the
pre-computed `traffic_light_series.parquet`. The full-period classification is retained as a separate "overall zone"
column but is labelled "full-period scaled, for illustration only".

---

## 11. GARCH-FHS vs. Monte Carlo Identical Hits (synthetic run) — Investigation (2026-10-25)
In the synthetic run, GARCH-FHS and Monte Carlo both showed 13 hits and identical p-values.
Investigation: The synthetic data had low volatility clustering, causing GARCH to produce conditional
volatility forecasts close to the unconditional variance. With similar mean VaR levels and the same
underlying synthetic return path, two well-calibrated models can coincidentally agree on which days
cross the threshold. The hit dates were verified to be identical in the synthetic run.
This was **coincidence on synthetic data, not a code bug** — confirmed by running both models on real data,
where they diverge by 11 separate exception dates. The GARCH failed-forecast count was 0 in both runs.

A diagnostic test was added to confirm GARCH and MC hit dates differ on real data.

---

## 12. ES Test Low Power Warning (2026-10-25)
The McNeil-Frey exceedance-residual test has low power when the number of tail exceedances is small
(typically < 30). With 1% VaR over 1,964 test-period days, expected exceedances ≈ 19.6, which is near the
lower bound for reliable inference. This limitation is now stated in the report's Limitations section.

---

## 13. Dynamic Warm-Up Slicing for Out-of-Sample Evaluation (2026-10-25)
In the initial implementation, `pipeline.py` statically sliced test data starting at `"2017-01-01"`. When the validation
grid search selected `opt_window = 1000` trading days, 1,000 trading days prior to 2019-01-02 reached back to `2015-01-12`.
Because `2017-01-01` provided only ~500 days of history prior to 2019-01-02, the first ~498 test-period days (2019 to late 2020)
were consumed by `run_walkforward`'s initial warm-up, causing the test period forecast outputs to begin on `2020-12-22` (1,466 obs).

**Fix applied**: Slicing for both validation and test periods in `pipeline.py` is now dynamic: it locates the exact integer index
of the period start date in the returns matrix and steps back exactly `opt_window` trading days into the preceding period.
As a result:
- **Training Period**: 1,910 returns (2007-06-04 to 2014-12-31)
- **Validation Period**: 1,006 returns (2015-01-02 to 2018-12-31)
- **Test Period**: 1,964 forecasts (2019-01-02 to 2026-10-23)
The entire test period is now evaluated from day one without truncation.

