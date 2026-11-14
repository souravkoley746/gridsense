"""Reference forecasts evaluated on the same timestamps as the ML models (FR06)."""
from __future__ import annotations
import pandas as pd

BASELINES = {
    "Persistence": "lag_1",
    "Daily seasonal-naive": "lag_24",
    "Weekly seasonal-naive": "lag_168",
}


def baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: frame[col] for name, col in BASELINES.items()}, index=frame.index)
