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
- **Core Dependencies**: Exact pinned versions in `requirements.txt` to ensure 100% reproducible environments across Windows, Linux, and CI:
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `scipy==1.13.1`
  - `statsmodels==0.14.2`
  - `arch==7.0.0`
  - `streamlit==1.36.0`
  - `plotly==5.22.0`
  - `pytest==8.2.2`
  - `pyyaml==6.0.1`
  - `yfinance==0.2.40`
  - `pyarrow==16.1.0`
  - `jinja2==3.1.4`

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
