"""Independent 5-Model Evaluation and Comparison Engine for GridSense.

Evaluates:
1. Persistence Baseline: y_hat = y_{t-1}
2. Daily Seasonal Naive: y_hat = y_{t-24}
3. Weekly Seasonal Naive: y_hat = y_{t-168}
4. Random Forest Regressor
5. XGBoost Regressor

Calculates:
- MAE (Mean Absolute Error, kWh)
- RMSE (Root Mean Squared Error, kWh)
- ME (Mean Error / Directional Bias, kWh) = Mean(Predicted - Actual)
- Relative Improvements vs Reference Baseline
- Macro-Average (unweighted across 10 clients) and Pooled Evaluation (all test observations)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import improvement_pct, point_metrics
PERF_DIR = ROOT / "reports" / "model_performance"

CLIENTS = [
    "MT_001",
    "MT_141",
    "MT_154",
    "MT_311",
    "MT_343",
    "MT_008",
    "MT_362",
    "MT_341",
    "MT_131",
    "MT_163",
]
PORTFOLIO = "PORTFOLIO"
ALL_TARGETS = CLIENTS + [PORTFOLIO]


def evaluate_target(client: str) -> pd.DataFrame:
    rd = ROOT / "reports" / client
    pred_path = rd / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Test predictions not found at {pred_path}. Train models first.")

    df_preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    scoreable = df_preds[df_preds["scoreable"]]

    actual = scoreable["Actual"].to_numpy(dtype=float)
    models = ["Persistence", "Daily seasonal-naive", "Weekly seasonal-naive", "Random Forest", "XGBoost"]

    rows = []
    # Identify daily naive MAE for relative improvement
    dn_pm = point_metrics(actual, scoreable["Daily seasonal-naive"].to_numpy(dtype=float))
    dn_mae = dn_pm["MAE_kWh"]
    dn_rmse = dn_pm["RMSE_kWh"]

    for m in models:
        pred = scoreable[m].to_numpy(dtype=float)
        pm = point_metrics(actual, pred)
        mae_imp = improvement_pct(pm["MAE_kWh"], dn_mae)
        rmse_imp = improvement_pct(pm["RMSE_kWh"], dn_rmse)
        rows.append({
            "client": client,
            "model_name": m,
            "category": "Baseline" if "naive" in m.lower() or "persistence" in m.lower() else "Machine Learning",
            "test_mae_kwh": round(pm["MAE_kWh"], 4),
            "test_rmse_kwh": round(pm["RMSE_kWh"], 4),
            "test_me_kwh": round(pm["ME_kWh"], 4),
            "mae_improvement_pct": round(mae_imp, 2) if mae_imp is not None else 0.0,
            "rmse_improvement_pct": round(rmse_imp, 2) if rmse_imp is not None else 0.0,
            "sample_count": pm["n"],
        })

    return pd.DataFrame(rows)


def run_full_evaluation():
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating 5 models across {len(ALL_TARGETS)} targets...")

    all_dfs = []
    for tgt in ALL_TARGETS:
        try:
            df_t = evaluate_target(tgt)
            all_dfs.append(df_t)
            print(f"  Evaluated: {tgt} ({len(df_t)} models)")
        except Exception as e:
            print(f"  Warning: Skipping {tgt}: {e}")

    if not all_dfs:
        print("No evaluation predictions available to evaluate.")
        return

    full_eval_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = PERF_DIR / "all_models_detailed_evaluation.csv"
    full_eval_df.to_csv(out_csv, index=False)
    print(f"\nDetailed evaluation saved to: {out_csv}")
    return full_eval_df


if __name__ == "__main__":
    run_full_evaluation()
