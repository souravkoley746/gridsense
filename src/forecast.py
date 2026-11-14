"""Forecast generation, multi-step recursive forecasting, and CSV export (Phases 4 & 6.1).

Integrity Checks:
- Model artefact SHA-256 validation
- Feature column order check
- Target unit verification (kWh)
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .data_loader import sha256_file
from .evaluate import classify_demand
from .features import FEATURE_COLUMNS, HOUR, InsufficientHistory, feature_row


class ForecastError(Exception):
    """Raised when a forecast cannot be generated reliably."""


def artefacts_exist(model_dir: Path | str) -> bool:
    d = Path(model_dir)
    return (d / "model.joblib").exists() and (d / "metadata.json").exists()


def load_artefacts(model_dir: Path | str):
    """Load model and metadata with integrity checks."""
    d = Path(model_dir)
    if not artefacts_exist(d):
        raise ForecastError(f"No trained model found in {d}. Run the training pipeline first.")
    with open(d / "metadata.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if meta.get("feature_columns") != FEATURE_COLUMNS:
        raise ForecastError(
            f"Metadata mismatch: saved feature order {meta.get('feature_columns')} differs from "
            f"expected {FEATURE_COLUMNS}. Inference blocked."
        )
    if meta.get("target_unit") != "kWh":
        raise ForecastError(
            f"Metadata mismatch: unexpected target unit {meta.get('target_unit')!r}. Inference blocked."
        )
    if sha256_file(d / "model.joblib") != meta.get("model_sha256"):
        raise ForecastError("Model file SHA-256 checksum mismatch: the artefact was corrupted or modified. Inference blocked.")

    model = joblib.load(d / "model.joblib")
    return model, meta


def _predict_one(model, row: dict) -> float:
    x = pd.DataFrame([row])[FEATURE_COLUMNS]
    val = float(model.predict(x)[0])
    if not np.isfinite(val):
        raise ForecastError("Model produced a non-finite prediction.")
    return max(val, 0.0)  # Energy cannot be negative


def recursive_forecast(
    model,
    history: pd.Series,
    horizon: int = 24,
    meta: dict | None = None,
    client: str | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Forecast `horizon` hours after the last observation using recursive multi-step forecasting.

    Strict leakage control: Only historical observations up to the forecast origin are used.
    Future predictions are recursively fed back into lag features without using actual future data.
    """
    if horizon < 1:
        raise ForecastError("Horizon must be at least 1 hour.")
    try:
        work = history.copy()
        issue = work.index[-1]
        rows = []
        for step in range(1, horizon + 1):
            target = issue + step * HOUR
            pred = _predict_one(model, feature_row(work, target))
            rows.append((issue, target, step, pred))
            work.loc[target] = pred
    except InsufficientHistory as exc:
        raise ForecastError(str(exc)) from exc

    df = pd.DataFrame(rows, columns=["issue_time", "target_time", "forecast_horizon_hours", "predicted_kwh"])
    df["unit"] = "kWh"
    df["client"] = client or (meta.get("client") if meta else "series")
    df["model_name"] = meta.get("selected_model", "Selected Model") if meta else "Model"
    df["model"] = df["model_name"]

    # Demand thresholds (derived strictly from training data)
    p25 = meta.get("p25_threshold_kwh", 0.0) if meta else 0.0
    p95 = threshold if threshold is not None else (meta.get("alert_threshold_kwh", 1e9) if meta else 1e9)

    tiers, explanations, alerts, statuses = [], [], [], []
    for pred in df["predicted_kwh"]:
        t, exp, al = classify_demand(pred, p25, p95)
        tiers.append(t)
        explanations.append(exp)
        alerts.append(al)
        statuses.append("ALERT" if al else "NO_ALERT")

    df["alert_level"] = tiers
    df["alert_status"] = statuses
    df["alert"] = alerts
    df["alert_explanation"] = explanations
    df["alert_threshold_kwh"] = p95

    # Prediction intervals if calibrated
    if meta is not None:
        q = meta.get("uncertainty", {})
        lo, hi = q.get("lower_offsets"), q.get("upper_offsets")
        if lo and hi:
            idx = np.minimum(df["forecast_horizon_hours"].to_numpy() - 1, len(lo) - 1)
            df["lower_kwh"] = np.maximum(df["predicted_kwh"] + np.asarray(lo)[idx], 0.0)
            df["upper_kwh"] = df["predicted_kwh"] + np.asarray(hi)[idx]

    return df


def forecast_next_hour(
    model,
    history: pd.Series,
    meta: dict | None = None,
    client: str | None = None,
) -> pd.Series:
    """Generate the genuine next-hour electricity consumption forecast."""
    return recursive_forecast(model, history, horizon=1, meta=meta, client=client).iloc[0]


def format_export_csv(fc_df: pd.DataFrame) -> str:
    """Export forecast results formatted according to Phase 6.1 schema."""
    cols = [
        "issue_time",
        "target_time",
        "forecast_horizon_hours",
        "client",
        "model_name",
        "predicted_kwh",
        "unit",
        "alert_status",
        "alert_level",
    ]
    export_cols = [c for c in cols if c in fc_df.columns]
    return fc_df[export_cols].to_csv(index=False)
