"""Custom exception hierarchy for the Risk Validation Workbench."""


class RVWError(Exception):
    """Base exception for all RVW errors."""
    pass


class DataValidationError(RVWError):
    """Raised when data integrity checks fail critically."""
    pass


class LookAheadLeakageError(RVWError):
    """Raised when data leakage across time periods or forecast dates is detected."""
    pass


class HyperparameterUnfrozenError(RVWError):
    """Raised when evaluation on test period is attempted before hyperparameter freeze."""
    pass


class ModelFitError(RVWError):
    """Raised when model estimation or calibration fails to converge."""
    pass


class ConfigurationError(RVWError):
    """Raised when configuration values are missing, invalid, or inconsistent."""
    pass
