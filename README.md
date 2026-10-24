# Market Risk Model Validation Workbench (RVW)

[![CI](https://github.com/randomaban-beep/risk-validation-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/randomaban-beep/risk-validation-workbench/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Regulatory & Production Disclaimer**:
> This software is an **educational prototype, inspired by model-validation practice; not a production risk system; no claim of compliance with Basel, SR 11-7, or any other regulation.**

---

## Overview
The **Market Risk Model Validation Workbench** is an open-source, modular Python testing and validation framework for market risk models. It features out-of-sample walk-forward estimation, backtesting (Kupiec POF, Christoffersen Independence & Conditional Coverage, Basel 250-day Traffic Light, McNeil-Frey ES test), historical and hypothetical stress testing, and Euler component risk attribution.

## Headline Validation Results
*Results pending initial model execution on benchmark universe.*

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python -m rvw.cli fetch-data --config config/default.yaml
python -m rvw.cli run --config config/default.yaml
python -m rvw.cli report --config config/default.yaml
streamlit run app/streamlit_app.py
```
