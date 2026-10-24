"""Basel Committee Traffic Light Framework for 99% VaR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BaselZoneResult:
    """Basel traffic light classification result."""
    zone: str  # "Green", "Yellow", "Red"
    exceptions: int
    window_length: int
    confidence_level: float
    cumulative_probability: float
    multiplier_penalty: float
    regulatory_interpretation: str
    disclaimer: str = "traffic-light classification, for illustration"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "zone": self.zone,
            "exceptions": self.exceptions,
            "window_length": self.window_length,
            "confidence_level": self.confidence_level,
            "cumulative_probability": float(self.cumulative_probability),
            "multiplier_penalty": float(self.multiplier_penalty),
            "regulatory_interpretation": self.regulatory_interpretation,
            "disclaimer": self.disclaimer,
        }


def classify_basel_zone(
    exceptions: int,
    window_length: int = 250,
    confidence_level: float = 0.99,
) -> BaselZoneResult:
    """Classify a 99% VaR model into Green, Yellow, or Red zone over a 250-day window.

    Standard Basel Classification:
    - Green (0 - 4 exceptions): No penalty (multiplier = 3.00). Cumulative binomial <= 0.89.
    - Yellow (5 - 9 exceptions): Supervisory review, capital scaling add-on (multiplier = 3.40 - 3.85).
    - Red (10+ exceptions): Presumption of an inaccurate model, automatic multiplier = 4.00.

    Args:
        exceptions: Count of realized VaR exceptions.
        window_length: Regulatory evaluation sample length (default 250 trading days).
        confidence_level: Target coverage level (0.99).

    Returns:
        BaselZoneResult: Structured status with regulatory interpretation.
    """
    p0 = 1.0 - confidence_level
    cum_prob = float(stats.binom.cdf(exceptions, window_length, p0))

    if exceptions <= 4:
        zone = "Green"
        penalty = 0.0
        interpretation = "Model accepted. No supervisory capital add-on warranted."
    elif exceptions <= 9:
        zone = "Yellow"
        # Standard Basel penalty tiers: 5 -> +0.40, 6 -> +0.50, 7 -> +0.65, 8 -> +0.75, 9 -> +0.85
        penalty_table = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        penalty = penalty_table.get(exceptions, 0.85)
        interpretation = "Supervisory scrutiny. Scaling factor add-on applies."
    else:
        zone = "Red"
        penalty = 1.00
        interpretation = "Model rejected. Severe underestimation of risk. Mandatory scaling factor 4.00."

    return BaselZoneResult(
        zone=zone,
        exceptions=exceptions,
        window_length=window_length,
        confidence_level=confidence_level,
        cumulative_probability=cum_prob,
        multiplier_penalty=penalty,
        regulatory_interpretation=interpretation,
    )


def compute_rolling_traffic_light(
    hits: pd.Series | np.ndarray,
    window: int = 250,
    confidence_level: float = 0.99,
) -> pd.DataFrame:
    """Compute rolling 250-day Basel traffic light zone time series.

    Args:
        hits: Binary hit series indexed by DatetimeIndex.
        window: Rolling observation window (default 250 days).
        confidence_level: VaR confidence level.

    Returns:
        pd.DataFrame: Time series with columns:
            ['Rolling_Exceptions', 'Zone', 'Multiplier_Penalty', 'Cumulative_Prob']
    """
    if isinstance(hits, np.ndarray):
        series = pd.Series(hits)
    else:
        series = hits.copy()

    rolling_exc = series.rolling(window=window).sum().dropna()

    records = []
    for date, count in rolling_exc.items():
        res = classify_basel_zone(int(count), window_length=window, confidence_level=confidence_level)
        records.append({
            "Rolling_Exceptions": int(count),
            "Zone": res.zone,
            "Multiplier_Penalty": res.multiplier_penalty,
            "Cumulative_Prob": res.cumulative_probability,
        })

    df = pd.DataFrame(records, index=rolling_exc.index)
    df.attrs["disclaimer"] = "traffic-light classification, for illustration"
    return df
