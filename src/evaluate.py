"""Evaluation metrics, bias calculations, and demand classification (Phase 3.4 & 3.7).

Calculates:
- MAE: Mean Absolute Error in kWh
- RMSE: Root Mean Squared Error in kWh
- ME: Mean Error in kWh = mean(predicted - actual)
  (positive = overprediction tendency, negative = underprediction tendency)
- Relative improvement (%) against reference baseline:
  Improvement (%) = ((Baseline Error - Model Error) / Baseline Error) * 100
- Multi-tier demand classification: LOW, NORMAL, HIGH
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    return a[ok], p[ok]


def point_metrics(actual, predicted) -> dict:
    """Calculate point forecast evaluation metrics on clean observations."""
    a, p = _clean(actual, predicted)
    if len(a) == 0:
        raise ValueError("Empty evaluation set: no valid actual and predicted pairs.")
    diff = p - a
    return {
        "MAE_kWh": float(np.mean(np.abs(diff))),
        "RMSE_kWh": float(np.sqrt(np.mean(diff ** 2))),
        "ME_kWh": float(np.mean(diff)),  # positive = overprediction, negative = underprediction
        "n": int(len(a)),
    }


def improvement_pct(model_error: float, baseline_error: float) -> float | None:
    """Relative improvement in error metric vs baseline: ((baseline - model) / baseline) * 100."""
    if baseline_error == 0:
        return None
    return float(100.0 * (1.0 - model_error / baseline_error))


def classify_demand(kwh: float, p25_threshold: float, p95_threshold: float) -> tuple[str, str, bool]:
    """Classify predicted electricity consumption into statistical demand tiers.

    Returns:
    --------
    tier : str ("LOW", "NORMAL", "HIGH")
    explanation : str
    is_alert : bool
    """
    if not np.isfinite(kwh):
        return "UNKNOWN", "Non-finite forecast value.", False
    if kwh > p95_threshold:
        return (
            "HIGH",
            f"Forecast ({kwh:.1f} kWh) exceeds historical training P95 threshold ({p95_threshold:.1f} kWh). "
            f"Statistical high-demand alert.",
            True,
        )
    elif kwh < p25_threshold:
        return (
            "LOW",
            f"Forecast ({kwh:.1f} kWh) is below historical training P25 threshold ({p25_threshold:.1f} kWh). "
            f"Substantially lower consumption expected.",
            False,
        )
    else:
        return (
            "NORMAL",
            f"Forecast ({kwh:.1f} kWh) is within typical operating range ({p25_threshold:.1f} to {p95_threshold:.1f} kWh).",
            False,
        )


def _div(n, d):
    return None if d == 0 else float(n / d)


def alert_metrics(actual, predicted, threshold: float) -> dict:
    """Calculate classification performance for the high-demand statistical alert."""
    a, p = _clean(actual, predicted)
    act = a > threshold
    alr = p > threshold
    tp = int((act & alr).sum())
    fp = int((~act & alr).sum())
    fn = int((act & ~alr).sum())
    tn = int((~act & ~alr).sum())

    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    f1 = None if (prec is None or rec is None or (prec + rec) == 0) else 2 * prec * rec / (prec + rec)

    out = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "actual_events": int(act.sum()),
        "threshold": float(threshold),
        "peak_MAE_kWh": float(np.mean(np.abs(a[act] - p[act]))) if act.any() else None,
        "largest_underprediction_kWh": float(np.max(a - p)) if len(a) else None,
    }
    return out


def error_by_hour(index: pd.DatetimeIndex, actual, predicted) -> pd.DataFrame:
    """Mean absolute error broken down by hour of the day."""
    df = pd.DataFrame(
        {"actual": np.asarray(actual, float), "pred": np.asarray(predicted, float)},
        index=index,
    )
    df["abs_err"] = (df["actual"] - df["pred"]).abs()
    out = df.groupby(df.index.hour)["abs_err"].mean().rename("MAE_kWh").to_frame()
    out.index.name = "hour"
    return out.reset_index()
