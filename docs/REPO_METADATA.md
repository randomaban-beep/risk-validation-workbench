# Repository Metadata

- **Project Name**: Market Risk Model Validation Workbench (RVW)
- **Target Repository**: `https://github.com/randomaban-beep/risk-validation-workbench`
- **Default Branch**: `main`
- **Current Version**: `v0.1.0`
- **License**: MIT License
- **Programming Language**: Python 3.11.9
- **Operating System Compatibility**: Windows 11 / 10, Linux (Ubuntu), macOS

---

## Technical Stack & Pinned Ecosystem

| Component | Library & Version | Purpose |
|---|---|---|
| Numerical Computation | `numpy==1.26.4` | Matrix algebra, Cholesky factorization, RNG |
| Time Series & DataFrames | `pandas==2.2.2` | Datetime index manipulation, return aggregation |
| Mathematical Statistics | `scipy==1.13.1` | Distribution fitting (Student-t, Normal), CDF, PPF |
| Statistical Models | `statsmodels==0.14.2` | Time series diagnostics, hypothesis testing |
| Volatility Modeling | `arch==7.0.0` | GARCH(1,1) Student-t innovations estimation |
| Interactive Dashboard | `streamlit==1.36.0` | Multi-page reactive validation dashboard |
| Visualizations | `plotly==5.22.0` | Dynamic charts (VaR envelopes, traffic lights, attribution) |
| Test Suite | `pytest==8.2.2` | Automated unit and integration testing |
| Data Caching | `pyarrow==16.1.0` | Parquet caching engine with JSON sidecar metadata |
| Configuration | `pyyaml==6.0.1` | YAML configuration file parser |
| Report Templating | `jinja2==3.1.4` | HTML and Markdown validation report rendering |
| Command-Line Interface | `click==8.1.7` | Modular CLI command structure |

---

## Directory Architecture

```
├── .github/workflows/ci.yml       # GitHub Actions automated test workflow
├── app/                           # Interactive Streamlit application
│   ├── pages/                     # Dedicated validation sub-pages (1-7)
│   └── streamlit_app.py           # Main dashboard interface
├── config/                        # YAML configuration profiles
│   ├── default.yaml               # Indicative universe default settings
│   └── synthetic_test.yaml        # Fast CI deterministic test profile
├── data/                          # Data directory
│   ├── cache/                     # Parquet price cache (gitignored)
│   └── sample/                    # Committed synthetic dataset fixture
├── docs/                          # Project documentation and sample reports
│   ├── REPO_METADATA.md
│   └── sample_report/             # Exported HTML and Markdown sample reports
├── reports/                       # Generated validation reports
├── results/                       # Computed pipeline result artifacts (JSON/parquet)
├── src/rvw/                       # Core python package
│   ├── attribution/               # Euler component VaR/ES and what-if rebalancing
│   ├── backtests/                 # Kupiec, Christoffersen, Basel, McNeil-Frey tests
│   ├── data/                      # Ingestion, caching, validation, synthetic data
│   ├── models/                    # Six standardized VaR/ES calculation models
│   ├── portfolio/                 # Portfolio weights and PeriodSplit freeze gate
│   ├── reporting/                 # Jinja2 report generator and rule-based findings
│   ├── stress/                    # Historical replay, hypothetical shocks, stressed covar
│   ├── walkforward/               # Strict out-of-sample rolling execution engine
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Pydantic/dataclass configuration loader
│   ├── exceptions.py              # Custom error hierarchy
│   ├── logging_utils.py           # Dual console + JSONL audit logging
│   ├── pipeline.py                # End-to-end execution pipeline
│   └── tuning.py                  # Hyperparameter selection and freeze manager
└── tests/                         # Comprehensive pytest suite (48+ unit tests)
```

---

## Regulatory & Production Disclaimer
This software is an **educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.**
