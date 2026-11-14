"""Leakage-controlled feature engineering (Phase 2).

Every feature for target hour t uses exclusively historical observations up to t - 1 hour.
Target values never enter their own lag or rolling statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]
MIN_HISTORY = 168
HOUR = pd.Timedelta(hours=1)


class InsufficientHistory(Exception):
    """Not enough (or misaligned) recent observations to build features."""


def make_supervised(hourly_energy: pd.Series) -> pd.DataFrame:
    """Construct supervised feature matrix from an hourly energy series.

    Strict leakage prevention:
    - Lag features use shift(1), shift(24), shift(168).
    - Rolling calculations are applied AFTER a shift(1), guaranteeing that
      the target observation at timestamp t is strictly excluded from rolling windows.
    """
    y = hourly_energy.sort_index().astype(float)
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("A DatetimeIndex is required.")
    if y.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")
    if not np.isfinite(y.dropna().to_numpy()).all():
        raise ValueError("Non-finite observations detected.")
    expected = pd.date_range(start=y.index.min(), end=y.index.max(), freq="h")
    if not y.index.equals(expected):
        raise ValueError("The series must have a continuous hourly index with no gaps.")

    frame = pd.DataFrame(index=y.index)
    frame["target_kwh"] = y
    frame["hour"] = frame.index.hour
    frame["day_of_week"] = frame.index.dayofweek
    frame["day_of_month"] = frame.index.day
    frame["month"] = frame.index.month
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)

    # Historical lags
    frame["lag_1"] = y.shift(1)
    frame["lag_24"] = y.shift(24)
    frame["lag_168"] = y.shift(168)

    # Shifted rolling windows: target observation at hour t never enters the window
    hist = y.shift(1)
    frame["rolling_mean_24"] = hist.rolling(24).mean()
    frame["rolling_std_24"] = hist.rolling(24).std()
    frame["rolling_mean_168"] = hist.rolling(168).mean()

    return frame.dropna(subset=FEATURE_COLUMNS + ["target_kwh"])


def feature_row(history: pd.Series, target_ts) -> dict:
    """Generate a single feature row for target hour `target_ts` from history ending at `target_ts - 1h`.

    Validates chronological alignment and ensures sufficient history (>= 168 hours).
    """
    t = pd.Timestamp(target_ts)
    if len(history) < MIN_HISTORY:
        raise InsufficientHistory(
            f"At least {MIN_HISTORY} hourly observations are required; received {len(history)}."
        )
    if history.index[-1] != t - HOUR:
        raise InsufficientHistory(
            f"History must end at {t - HOUR} to forecast target {t}; it ends at {history.index[-1]}."
        )

    tail = history.iloc[-MIN_HISTORY:].to_numpy(dtype=float)
    if not np.isfinite(tail).all():
        raise InsufficientHistory("Recent history contains missing or non-finite observations.")

    return {
        "hour": t.hour,
        "day_of_week": t.dayofweek,
        "day_of_month": t.day,
        "month": t.month,
        "is_weekend": int(t.dayofweek >= 5),
        "lag_1": float(tail[-1]),
        "lag_24": float(tail[-24]),
        "lag_168": float(tail[-168]),
        "rolling_mean_24": float(tail[-24:].mean()),
        "rolling_std_24": float(tail[-24:].std(ddof=1)),
        "rolling_mean_168": float(tail.mean()),
    }
