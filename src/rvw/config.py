"""Configuration loading and validation module for the Risk Validation Workbench."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class UniverseConfig:
    """Configuration for indicative universe and download date range."""
    tickers: List[str]
    start_date: str
    end_date: Optional[str] = None


@dataclass
class DateRangeConfig:
    """A start-and-end date range specification."""
    start_date: str
    end_date: Optional[str] = None


@dataclass
class PeriodsConfig:
    """Time periods partition configuration."""
    training: DateRangeConfig
    validation: DateRangeConfig
    test: DateRangeConfig


@dataclass
class RiskConventionsConfig:
    """Risk conventions for VaR and ES estimation."""
    horizon_days: int = 1
    var_alphas: List[float] = field(default_factory=lambda: [0.95, 0.99])
    es_alphas: List[float] = field(default_factory=lambda: [0.975, 0.99])


@dataclass
class ModelsConfig:
    """Hyperparameters and configuration for each risk model."""
    historical_simulation: Dict[str, Any] = field(default_factory=lambda: {"interpolation": "linear"})
    parametric_normal: Dict[str, Any] = field(default_factory=lambda: {"assume_zero_mean": False})
    parametric_t: Dict[str, Any] = field(default_factory=lambda: {"fit_method": "mle", "min_df": 2.1})
    ewma: Dict[str, Any] = field(default_factory=lambda: {"lambda_param": 0.94, "lambda_candidates": [0.94, 0.97]})
    garch_fhs: Dict[str, Any] = field(default_factory=lambda: {"p": 1, "q": 1, "dist": "StudentsT", "resample_size": 10000})
    monte_carlo: Dict[str, Any] = field(default_factory=lambda: {"num_simulations": 25000, "distribution": "normal"})


@dataclass
class WalkforwardConfig:
    """Configuration for walk-forward out-of-sample engine."""
    window_length: int = 500
    window_candidates: List[int] = field(default_factory=lambda: [250, 500, 1000])
    window_type: str = "rolling"
    refit_every: int = 5
    fallback_to_historical_on_failure: bool = True


@dataclass
class SeedsConfig:
    """RNG seeds for deterministic execution."""
    global_seed: int = 42
    monte_carlo_seed: int = 1337
    garch_seed: int = 2024
    synthetic_seed: int = 777


@dataclass
class DataValidationConfig:
    """Thresholds for price and return sanity checking."""
    max_return_jump: float = 0.25
    max_stale_days: int = 5
    max_calendar_gap_days: int = 5


@dataclass
class PathsConfig:
    """Path conventions for data, results, reports, and logs."""
    cache_dir: str = "data/cache"
    sample_data_file: str = "data/sample/synthetic_prices.parquet"
    results_dir: str = "results"
    reports_dir: str = "reports"
    logs_dir: str = "logs"


@dataclass
class AppConfig:
    """Root configuration object for the workbench."""
    universe: UniverseConfig
    periods: PeriodsConfig
    freeze_hyperparameters: bool = False
    frozen_params_file: str = "results/frozen_hyperparameters.json"
    risk_conventions: RiskConventionsConfig = field(default_factory=RiskConventionsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    walkforward: WalkforwardConfig = field(default_factory=WalkforwardConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    data_validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    portfolios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scenarios: Dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def load_config(config_path: str | Path = "config/default.yaml") -> AppConfig:
    """Load configuration from YAML file with validation and default fallback.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig: Instantiated and validated configuration object.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If YAML parsing fails or mandatory keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected a dictionary mapping.")

    # Universe
    u_dict = raw.get("universe", {})
    universe = UniverseConfig(
        tickers=list(u_dict.get("tickers", [])),
        start_date=str(u_dict.get("start_date", "2007-06-01")),
        end_date=u_dict.get("end_date"),
    )

    # Periods
    p_dict = raw.get("periods", {})
    periods = PeriodsConfig(
        training=DateRangeConfig(
            start_date=p_dict.get("training", {}).get("start_date", "2007-06-01"),
            end_date=p_dict.get("training", {}).get("end_date", "2014-12-31"),
        ),
        validation=DateRangeConfig(
            start_date=p_dict.get("validation", {}).get("start_date", "2015-01-01"),
            end_date=p_dict.get("validation", {}).get("end_date", "2018-12-31"),
        ),
        test=DateRangeConfig(
            start_date=p_dict.get("test", {}).get("start_date", "2019-01-01"),
            end_date=p_dict.get("test", {}).get("end_date"),
        ),
    )

    # Risk Conventions
    rc_dict = raw.get("risk_conventions", {})
    risk_conventions = RiskConventionsConfig(
        horizon_days=int(rc_dict.get("horizon_days", 1)),
        var_alphas=[float(x) for x in rc_dict.get("var_alphas", [0.95, 0.99])],
        es_alphas=[float(x) for x in rc_dict.get("es_alphas", [0.975, 0.99])],
    )

    # Models
    m_dict = raw.get("models", {})
    models = ModelsConfig(
        historical_simulation=m_dict.get("historical_simulation", {"interpolation": "linear"}),
        parametric_normal=m_dict.get("parametric_normal", {"assume_zero_mean": False}),
        parametric_t=m_dict.get("parametric_t", {"fit_method": "mle", "min_df": 2.1}),
        ewma=m_dict.get("ewma", {"lambda_param": 0.94, "lambda_candidates": [0.94, 0.97]}),
        garch_fhs=m_dict.get("garch_fhs", {"p": 1, "q": 1, "dist": "StudentsT", "resample_size": 10000}),
        monte_carlo=m_dict.get("monte_carlo", {"num_simulations": 25000, "distribution": "normal"}),
    )

    # Walkforward
    wf_dict = raw.get("walkforward", {})
    walkforward = WalkforwardConfig(
        window_length=int(wf_dict.get("window_length", 500)),
        window_candidates=[int(w) for w in wf_dict.get("window_candidates", [250, 500, 1000])],
        window_type=str(wf_dict.get("window_type", "rolling")),
        refit_every=int(wf_dict.get("refit_every", 5)),
        fallback_to_historical_on_failure=bool(wf_dict.get("fallback_to_historical_on_failure", True)),
    )

    # Seeds
    s_dict = raw.get("seeds", {})
    seeds = SeedsConfig(
        global_seed=int(s_dict.get("global_seed", 42)),
        monte_carlo_seed=int(s_dict.get("monte_carlo_seed", 1337)),
        garch_seed=int(s_dict.get("garch_seed", 2024)),
        synthetic_seed=int(s_dict.get("synthetic_seed", 777)),
    )

    # Data Validation
    dv_dict = raw.get("data_validation", {})
    data_validation = DataValidationConfig(
        max_return_jump=float(dv_dict.get("max_return_jump", 0.25)),
        max_stale_days=int(dv_dict.get("max_stale_days", 5)),
        max_calendar_gap_days=int(dv_dict.get("max_calendar_gap_days", 5)),
    )

    # Paths
    pa_dict = raw.get("paths", {})
    paths = PathsConfig(
        cache_dir=str(pa_dict.get("cache_dir", "data/cache")),
        sample_data_file=str(pa_dict.get("sample_data_file", "data/sample/synthetic_prices.parquet")),
        results_dir=str(pa_dict.get("results_dir", "results")),
        reports_dir=str(pa_dict.get("reports_dir", "reports")),
        logs_dir=str(pa_dict.get("logs_dir", "logs")),
    )

    return AppConfig(
        universe=universe,
        periods=periods,
        freeze_hyperparameters=bool(raw.get("freeze_hyperparameters", False)),
        frozen_params_file=str(raw.get("frozen_params_file", "results/frozen_hyperparameters.json")),
        risk_conventions=risk_conventions,
        models=models,
        walkforward=walkforward,
        seeds=seeds,
        data_validation=data_validation,
        portfolios=raw.get("portfolios", {}),
        scenarios=raw.get("scenarios", {}),
        paths=paths,
    )
