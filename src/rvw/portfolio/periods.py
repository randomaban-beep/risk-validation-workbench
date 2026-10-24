"""Time period management, dataset splitting, and hyperparameter freeze gatekeeper."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from rvw.config import AppConfig, PeriodsConfig
from rvw.exceptions import HyperparameterUnfrozenError
from rvw.logging_utils import log_event

logger = logging.getLogger(__name__)


class PeriodSplit:
    """Partitions time series into Training/Warm-up, Validation, and Test periods.

    Enforces strict access control over the Test period to prevent data snooping
    and p-hacking prior to final model locking.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        config: AppConfig,
    ) -> None:
        """Initialize PeriodSplit with data and configuration.

        Args:
            data: Time series indexed by DatetimeIndex (prices or returns).
            config: Application configuration containing period boundaries and freeze flag.
        """
        self.data = data
        self.config = config
        self._test_evaluation_count = 0
        self._audit_log: list[Dict[str, Any]] = []

    @property
    def training_range(self) -> Tuple[str, Optional[str]]:
        """Return (start_date, end_date) for training period."""
        return (
            self.config.periods.training.start_date,
            self.config.periods.training.end_date,
        )

    @property
    def validation_range(self) -> Tuple[str, Optional[str]]:
        """Return (start_date, end_date) for validation period."""
        return (
            self.config.periods.validation.start_date,
            self.config.periods.validation.end_date,
        )

    @property
    def test_range(self) -> Tuple[str, Optional[str]]:
        """Return (start_date, end_date) for test period."""
        return (
            self.config.periods.test.start_date,
            self.config.periods.test.end_date,
        )

    def slice_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame | pd.Series:
        """Slice internal data by date boundaries.

        Args:
            start_date: Inclusive start date string (YYYY-MM-DD).
            end_date: Inclusive end date string (YYYY-MM-DD) or None.

        Returns:
            pd.DataFrame | pd.Series: Sliced subset.
        """
        subset = self.data.loc[start_date:]
        if end_date is not None:
            subset = subset.loc[:end_date]
        return subset

    def get_training(self) -> pd.DataFrame | pd.Series:
        """Retrieve Training / Warm-up period slice."""
        s, e = self.training_range
        return self.slice_data(s, e)

    def get_validation(self) -> pd.DataFrame | pd.Series:
        """Retrieve Validation period slice for model selection."""
        s, e = self.validation_range
        return self.slice_data(s, e)

    def get_test(self, caller_context: str = "walkforward_test") -> pd.DataFrame | pd.Series:
        """Retrieve Test period slice, guarded by the hyperparameter freeze protocol.

        Raises:
            HyperparameterUnfrozenError: If config.freeze_hyperparameters is False
                                         or if frozen_params_file does not exist.

        Args:
            caller_context: Identifier of caller requesting test access for auditing.

        Returns:
            pd.DataFrame | pd.Series: Sliced test dataset.
        """
        frozen_file = Path(self.config.frozen_params_file)

        if not self.config.freeze_hyperparameters:
            raise HyperparameterUnfrozenError(
                "Access to the Test period is locked: config.freeze_hyperparameters is False. "
                "You must benchmark models on the Validation period and freeze hyperparameters first."
            )

        if not frozen_file.exists():
            raise HyperparameterUnfrozenError(
                f"Access to the Test period is locked: frozen parameters artifact '{frozen_file}' "
                "was not found on disk. Save the frozen hyperparameters before evaluating test data."
            )

        # Increment evaluation audit count
        self._test_evaluation_count += 1
        now_utc = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        audit_entry = {
            "evaluation_index": self._test_evaluation_count,
            "timestamp": now_utc,
            "caller": caller_context,
            "frozen_file": str(frozen_file.resolve()),
        }
        self._audit_log.append(audit_entry)

        log_event(
            logger=logger,
            level=logging.INFO,
            event_type="TEST_PERIOD_ACCESSED",
            message=f"Test period accessed (count={self._test_evaluation_count}) by {caller_context}",
            **audit_entry,
        )

        s, e = self.test_range
        return self.slice_data(s, e)

    @property
    def test_evaluation_count(self) -> int:
        """Number of times the test period has been accessed."""
        return self._test_evaluation_count

    @property
    def audit_log(self) -> list[Dict[str, Any]]:
        """Audit trail of all test evaluations."""
        return list(self._audit_log)
