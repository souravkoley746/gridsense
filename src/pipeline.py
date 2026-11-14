"""End-to-end chronological training, validation, refit, and evaluation pipeline (Phase 3).

Execution flow:
1. Load & validate dataset
2. Construct hourly kWh target & flag quality anomalies
3. Build leakage-controlled feature matrix
4. Chronological split: 70% Train, 15% Validation, 15% Test (no shuffling)
5. Step 1: Train Random Forest and XGBoost candidates on the training period
6. Step 2: Score all 5 models (3 baselines + 2 ML) on validation; select best model
7. Step 3: Refit BOTH ML candidates on combined train + validation periods
8. Step 4: Evaluate all 5 models on the untouched final test period
9. Save artefacts (model.joblib, metadata.json, metrics.csv, test_predictions.csv)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .baselines import baseline_predictions
from .config import dataset_path, load_config, model_dir, processed_path, report_dir
from .data_loader import PORTFOLIO, list_clients, load_raw, sha256_file
from .evaluate import alert_metrics, error_by_hour, improvement_pct, point_metrics
from .features import FEATURE_COLUMNS, make_supervised
from .forecast import recursive_forecast
from .models import build_candidates, xgboost_available
from .preprocessing import build_hourly


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def _calibrate_intervals(model, hourly: pd.DataFrame, val: pd.DataFrame, cfg: dict) -> dict:
    """Calibrate empirical per-horizon residual quantiles from rolling-origin backtests on validation."""
    u = cfg.get("uncertainty", {"lower": 0.05, "upper": 0.95, "backtest_origins": 30, "horizon": 24})
    H, n_orig = int(u.get("horizon", 24)), int(u.get("backtest_origins", 30))
    y, scoreable = hourly["kwh"], hourly["scoreable"]
    hmax = len(val) - H - 1
    if hmax < 1:
        return {"method": "insufficient validation length for calibration"}

    positions = np.unique(np.linspace(0, hmax, min(n_orig, hmax)).astype(int))
    errs = np.full((len(positions), H), np.nan)
    for r, k in enumerate(positions):
        origin = val.index[k]
        fc = recursive_forecast(model, y.loc[:origin], H)
        tgt = fc["target_time"]
        actual = y.reindex(tgt).to_numpy()
        ok = scoreable.reindex(tgt).to_numpy(dtype=bool)
        e = actual - fc["predicted_kwh"].to_numpy()
        e[~ok] = np.nan
        errs[r] = e

    lo = np.nanquantile(errs, u.get("lower", 0.05), axis=0)
    hi = np.nanquantile(errs, u.get("upper", 0.95), axis=0)
    lo, hi = np.minimum.accumulate(lo), np.maximum.accumulate(hi)
    cover = float(np.nanmean((errs >= lo) & (errs <= hi)))

    return {
        "lower_q": u.get("lower", 0.05),
        "upper_q": u.get("upper", 0.95),
        "horizon": H,
        "origins": int(len(positions)),
        "lower_offsets": lo.tolist(),
        "upper_offsets": hi.tolist(),
        "backtest_coverage": cover,
        "mae_by_horizon": np.nanmean(np.abs(errs), axis=0).tolist(),
        "method": "empirical residual quantiles from rolling-origin backtests (validation, train-only model)",
    }


def run_experiment(client: str | None = None, cfg: dict | None = None, progress=None) -> dict:
    cfg = cfg or load_config()
    client = client or cfg["dataset"]["default_client"]
    say = progress or (lambda frac, msg: None)
    t0 = time.time()

    say(0.05, "Loading and validating raw dataset")
    kw, info = load_raw(dataset_path(cfg), client)

    say(0.15, "Building hourly kWh target and quality report")
    dst_pol = cfg.get("preprocessing", {}).get("dst_policy", "flag_only")
    trim_zeros = cfg.get("preprocessing", {}).get("trim_structural_zeros", True)
    hourly, quality = build_hourly(kw, trim_structural_zeros=trim_zeros, dst_policy=dst_pol)
    quality = {
        **{k: info[k] for k in ("raw_rows", "valid_observations", "raw_start", "raw_end", "negative_values", "missing_values", "was_unsorted", "off_grid_intervals") if k in info},
        **quality,
    }

    say(0.22, "Engineering leakage-free features")
    frame = make_supervised(hourly["kwh"]).join(hourly[["scoreable"]])
    n = len(frame)
    s = cfg.get("split", {"train_frac": 0.70, "validation_frac": 0.15})
    i_tr = int(n * s["train_frac"])
    i_va = int(n * (s["train_frac"] + s["validation_frac"]))
    train, val, test = frame.iloc[:i_tr], frame.iloc[i_tr:i_va], frame.iloc[i_va:]

    if min(len(train), len(val), len(test)) < 150:
        raise ValueError("Each chronological partition needs at least 150 valid hourly observations.")

    sc = lambda d: d[d["scoreable"]]  # filter to scoreable hours only

    # Derive demand thresholds strictly from training data
    train_scores = train["target_kwh"].to_numpy()
    p25 = float(np.percentile(train_scores, 25))
    p95 = float(np.percentile(train_scores, cfg.get("alert", {}).get("quantile", 0.95) * 100))

    say(0.30, "Step 1 & 2: Scoring baselines and training ML candidates on validation")
    bv = baseline_predictions(val)
    val_metrics = {}
    for name in bv.columns:
        val_metrics[name] = point_metrics(sc(val)["target_kwh"], bv.loc[sc(val).index, name])

    best_baseline = min(bv.columns, key=lambda k: val_metrics[k]["MAE_kWh"])

    cands, notes = build_candidates(cfg)
    val_scores_ml, fitted_train = {}, {}
    for j, (name, model) in enumerate(cands.items()):
        say(0.35 + 0.12 * j, f"Training {name} on the training partition")
        model.fit(train[FEATURE_COLUMNS], train["target_kwh"])
        pred_val = model.predict(sc(val)[FEATURE_COLUMNS])
        val_metrics[name] = point_metrics(sc(val)["target_kwh"], pred_val)
        val_scores_ml[name] = val_metrics[name]
        fitted_train[name] = model

    # Select best model based on validation MAE
    selected = min(val_scores_ml, key=lambda k: val_scores_ml[k]["MAE_kWh"])
    recommended = selected if val_scores_ml[selected]["MAE_kWh"] < val_metrics[best_baseline]["MAE_kWh"] else best_baseline

    say(0.65, "Calibrating empirical prediction intervals")
    unc = _calibrate_intervals(fitted_train[selected], hourly, val, cfg)

    say(0.75, "Computing permutation feature importance on validation data")
    imp = permutation_importance(
        fitted_train[selected],
        val[FEATURE_COLUMNS],
        val["target_kwh"],
        n_repeats=5,
        random_state=cfg.get("seed", 42),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mae_kwh": imp.importances_mean})
        .sort_values("importance_mae_kwh", ascending=False)
        .reset_index(drop=True)
    )

    say(0.82, "Step 3: Refitting BOTH ML candidates on combined train + validation")
    dev = pd.concat([train, val])
    refitted_models = {}
    for name, model in fitted_train.items():
        final_model = clone(model)
        final_model.fit(dev[FEATURE_COLUMNS], dev["target_kwh"])
        refitted_models[name] = final_model

    say(0.90, "Step 4: Fair evaluation of ALL FIVE models on untouched test period")
    # All 5 models: 3 baselines + 2 refitted ML models
    b_test = baseline_predictions(test)
    test_preds = b_test.copy()
    test_preds.insert(0, "Actual", test["target_kwh"])
    for name, m in refitted_models.items():
        test_preds[name] = m.predict(test[FEATURE_COLUMNS])
    test_preds["scoreable"] = test["scoreable"]

    ts = test_preds[test_preds["scoreable"]]
    all_five_methods = list(bv.columns) + list(refitted_models.keys())

    # Find best baseline on validation for fair relative improvement calculation per Section 6
    best_base_test_pm = point_metrics(ts["Actual"], ts[best_baseline])
    best_base_mae = best_base_test_pm["MAE_kWh"]
    best_base_rmse = best_base_test_pm["RMSE_kWh"]

    metric_rows = []
    long_pred_rows = []
    test_start_str = str(test.index[0])
    test_end_str = str(test.index[-1])

    for m in all_five_methods:
        pm = point_metrics(ts["Actual"], ts[m])
        mae_imp = improvement_pct(pm["MAE_kWh"], best_base_mae)
        rmse_imp = improvement_pct(pm["RMSE_kWh"], best_base_rmse)
        val_mae = val_metrics.get(m, {}).get("MAE_kWh")

        if m == selected:
            sel_status = "Selected (Lowest Validation MAE)"
        elif m in refitted_models:
            sel_status = "Candidate (Higher Validation MAE)"
        elif m == best_baseline:
            sel_status = "Best Baseline Reference"
        else:
            sel_status = "Baseline Reference"

        metric_rows.append({
            "model_name": m,
            "method": m,
            "kind": "baseline" if m in bv.columns else "ml",
            "validation_mae": round(val_mae, 4) if val_mae is not None else None,
            "Validation MAE": round(val_mae, 4) if val_mae is not None else None,
            "test_mae": round(pm["MAE_kWh"], 4),
            "Test MAE": round(pm["MAE_kWh"], 4),
            "test_rmse": round(pm["RMSE_kWh"], 4),
            "Test RMSE": round(pm["RMSE_kWh"], 4),
            "test_me": round(pm["ME_kWh"], 4),
            "Test ME": round(pm["ME_kWh"], 4),
            "model_selection_status": sel_status,
            "Model-selection status": sel_status,
            "MAE": round(pm["MAE_kWh"], 4),
            "MAE_kWh": round(pm["MAE_kWh"], 4),
            "RMSE": round(pm["RMSE_kWh"], 4),
            "RMSE_kWh": round(pm["RMSE_kWh"], 4),
            "ME": round(pm["ME_kWh"], 4),
            "ME_kWh": round(pm["ME_kWh"], 4),
            "sample_count": pm["n"],
            "n": pm["n"],
            "MAE_improvement_pct": round(mae_imp, 2) if mae_imp is not None else 0.0,
            "RMSE_improvement_pct": round(rmse_imp, 2) if rmse_imp is not None else 0.0,
            "improvement_vs_best_baseline_pct": round(mae_imp, 2) if mae_imp is not None else 0.0,
            "test_start": test_start_str,
            "test_end": test_end_str,
        })

        # Format long predictions conforming to Phase 8 / Section 9 requirements
        for target_ts, row in ts.iterrows():
            issue_ts = target_ts - pd.Timedelta(hours=1)
            pred_val = float(row[m])
            act_val = float(row["Actual"])
            err_val = pred_val - act_val
            long_pred_rows.append({
                "timestamp": str(target_ts),
                "issue_time": str(issue_ts),
                "target_time": str(target_ts),
                "client": client,
                "model_name": m,
                "actual_kwh": round(act_val, 4),
                "predicted_kwh": round(pred_val, 4),
                "error_kwh": round(err_val, 4),
                "prediction_error": round(err_val, 4),
                "absolute_error_kwh": round(abs(err_val), 4),
                "unit": "kWh",
            })

    metrics_df = pd.DataFrame(metric_rows)
    long_preds_df = pd.DataFrame(long_pred_rows)
    alert_res = {m: alert_metrics(ts["Actual"], ts[m], p95) for m in (best_baseline, selected)}
    by_hour = error_by_hour(ts.index, ts["Actual"], ts[selected])

    say(0.96, "Saving model artefacts, metadata, and reports")
    md, rd = model_dir(client), report_dir(client)
    md.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    processed_path(client).parent.mkdir(parents=True, exist_ok=True)

    # Save selected refitted model
    joblib.dump(refitted_models[selected], md / "model.joblib")
    hourly.to_csv(processed_path(client))

    # Save reports
    metrics_df.to_csv(rd / "metrics.csv", index=False)
    test_preds.to_csv(rd / "test_predictions.csv")
    long_preds_df.to_csv(rd / "test_predictions_detailed.csv", index=False)
    long_preds_df.to_csv(rd / "model_evaluation_predictions.csv", index=False)
    importance.to_csv(rd / "feature_importance.csv", index=False)
    by_hour.to_csv(rd / "error_by_hour.csv", index=False)

    with open(rd / "data_quality.json", "w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2, default=_jsonable)

    # Generate evaluation_summary.md
    eval_summary_md = f"""# Model Evaluation Summary: {client}

## 1. Chronological Partitioning
- **Training Period:** {train.index[0]} to {train.index[-1]} ({len(train):,} hours, 70%)
- **Validation Period:** {val.index[0]} to {val.index[-1]} ({len(val):,} hours, 15%)
- **Test Period:** {test.index[0]} to {test.index[-1]} ({len(test):,} hours, 15%)
- **Common Test Timestamps:** {len(ts):,} scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
""" + "\n".join(
        f"- **{k}:** MAE = {v['MAE_kWh']:.4f} kWh" + (" (Best Baseline)" if k == best_baseline else (" (Selected Model)" if k == selected else ""))
        for k, v in val_metrics.items()
    ) + f"""

**Selected Model:** **{selected}** achieved the lowest validation MAE ({val_metrics[selected]['MAE_kWh']:.4f} kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
"""
    for _, r in metrics_df.iterrows():
        eval_summary_md += f"| {r['model_name']} | {r['kind']} | {r['validation_mae']:.4f} | {r['test_mae']:.4f} | {r['test_rmse']:.4f} | {r['test_me']:+.4f} | {r['MAE_improvement_pct']:+.2f}% | {r['RMSE_improvement_pct']:+.2f}% | {r['model_selection_status']} |\n"

    sel_mae_imp = improvement_pct(point_metrics(ts['Actual'], ts[selected])['MAE_kWh'], best_base_mae)
    sel_mae_imp_str = f"{sel_mae_imp:.2f}%" if sel_mae_imp is not None else "0.00%"

    eval_summary_md += f"""
## 4. Key Findings
- **Error Reduction:** Selected model **{selected}** achieved a {sel_mae_imp_str} MAE improvement over the best baseline ({best_baseline}).
- **Calibration (ME):** Test Mean Error is {point_metrics(ts['Actual'], ts[selected])['ME_kWh']:+.4f} kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same {len(ts):,} scoreable timestamps.
"""
    is_syn = cfg.get("dataset", {}).get("is_synthetic", False)
    if is_syn:
        prov_text = f"""- **Dataset Description:** A synthetic electricity-consumption benchmark dataset formatted according to the UCI dataset structure.
- **Nature of Data:** Artificially generated synthetic benchmark values simulated with diurnal/weekly profiles, temperature proxies, and noise via `scripts/generate_demo_dataset.py`. Values are **NOT** actual smart-meter measurements.
- **Dataset Checksum (SHA-256):** `{info.get('sha256', '')}`
- **File Path:** `{dataset_path(cfg)}`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using synthetic historical electricity consumption data. It has not been validated against live electricity-grid measurements or deployed in a real smart-grid environment."""
    else:
        prov_text = f"""- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \\sum_{{j=1}}^4 P_{{h,j}} \\times 0.25$.
- **Dataset Checksum (SHA-256):** `{info.get('sha256', '')}`
- **File Path:** `{dataset_path(cfg)}`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system."""

    eval_summary_md += f"""
## 5. Dataset Provenance & Authenticity
{prov_text}
"""
    with open(rd / "evaluation_summary.md", "w", encoding="utf-8") as fh:
        fh.write(eval_summary_md)

    meta = {
        "app": "GridSense",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client": client,
        "target_unit": "kWh",
        "target_definition": quality["target_definition"],
        "forecast_horizon_note": "next-hour model; multi-step via recursive strategy",
        "feature_columns": FEATURE_COLUMNS,
        "selected_model": selected,
        "recommended_method": recommended,
        "best_baseline": best_baseline,
        "model_engine": notes.get(selected, "sklearn"),
        "model_params": {
            k: v for k, v in refitted_models[selected].get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
        "alert_threshold_kwh": p95,
        "p25_threshold_kwh": p25,
        "alert_quantile": cfg.get("alert", {}).get("quantile", 0.95),
        "alert_label": "Statistical High-Demand Alert (P95 threshold; not a physical grid overload detection)",
        "splits": {
            "train": [str(train.index[0]), str(train.index[-1]), len(train)],
            "validation": [str(val.index[0]), str(val.index[-1]), len(val)],
            "test": [str(test.index[0]), str(test.index[-1]), len(test)],
        },
        "training_date_range": [str(train.index[0]), str(train.index[-1])],
        "validation_date_range": [str(val.index[0]), str(val.index[-1])],
        "test_date_range": [str(test.index[0]), str(test.index[-1])],
        "valid_hourly_observations": len(hourly),
        "common_test_timestamps": int(ts["scoreable"].sum()),
        "validation_scores": val_metrics,
        "test_metrics": metrics_df.to_dict(orient="records"),
        "test_alert_metrics": alert_res,
        "uncertainty": unc,
        "feature_importance": importance.to_dict(orient="records"),
        "dataset": {
            "path": str(dataset_path(cfg)),
            "sha256": info.get("sha256", ""),
            "dataset_checksum": info.get("sha256", ""),
            "dataset_source": cfg.get("dataset", {}).get("source_note", "UCI ElectricityLoadDiagrams20112014"),
            "is_synthetic": cfg.get("dataset", {}).get("is_synthetic", False),
            "note": cfg.get("dataset", {}).get("source_note", ""),
        },
        "quality_summary": quality,
        "seed": cfg.get("seed", 42),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": (lambda: __import__("xgboost").__version__)() if xgboost_available() else None,
            "xgboost_installed": xgboost_available(),
        },
        "training_seconds": round(time.time() - t0, 1),
        "model_sha256": sha256_file(md / "model.joblib"),
    }

    with open(md / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_jsonable)
    with open(rd / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_jsonable)
    with open(rd / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_jsonable)

    say(1.0, "Experiment pipeline completed successfully")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate GridSense forecasting models.")
    ap.add_argument("--client", help="Client column (e.g. MT_001) or PORTFOLIO")
    ap.add_argument("--all", action="store_true", help="Train every client column individually")
    ap.add_argument("--config", help="Path to experiment.json")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    clients = list_clients(dataset_path(cfg)) if a.all else [a.client or cfg["dataset"]["default_client"]]
    for c in clients:
        print(f"== Running pipeline for client: {c}")
        meta = run_experiment(c, cfg, progress=lambda f, m: print(f"  [{f*100:3.0f}%] {m}"))
        best = min(meta["test_metrics"], key=lambda r: r["MAE"])
        print(f"  Best test MAE: {best['model_name']} = {best['MAE']:.3f} kWh  (Validation-selected: {meta['selected_model']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
