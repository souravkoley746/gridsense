"""Automated tests for forecasting engine, demand tiering, CSV export, and integrity checks (Phase 7.5)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

import src.config as config
from src.evaluate import classify_demand
from src.forecast import (
    ForecastError,
    format_export_csv,
    load_artefacts,
    recursive_forecast,
)


class TestForecastEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use existing trained MT_001 model in workspace
        cls.model_dir = config.ROOT / "models" / "MT_001"
        cls.hourly_path = config.processed_path("MT_001")
        if not (cls.model_dir / "model.joblib").exists():
            from src import pipeline
            pipeline.run_experiment("MT_001")

        cls.model, cls.meta = load_artefacts(cls.model_dir)
        cls.hourly = pd.read_csv(cls.hourly_path, index_col=0, parse_dates=True)["kwh"]

    def test_tc11_model_loading_and_next_hour_forecast(self):
        fc = recursive_forecast(self.model, self.hourly, horizon=1, meta=self.meta)
        self.assertEqual(len(fc), 1)
        row = fc.iloc[0]
        self.assertEqual(row["forecast_horizon_hours"], 1)
        self.assertEqual(row["target_time"], self.hourly.index[-1] + pd.Timedelta(hours=1))
        self.assertTrue(np.isfinite(row["predicted_kwh"]))
        self.assertGreaterEqual(row["predicted_kwh"], 0.0)

    def test_tc11b_multi_step_recursive_forecast(self):
        horizon = 24
        fc = recursive_forecast(self.model, self.hourly, horizon=horizon, meta=self.meta)
        self.assertEqual(len(fc), horizon)
        self.assertTrue(np.isfinite(fc["predicted_kwh"]).all())
        # Prediction intervals sanity
        if "lower_kwh" in fc.columns and "upper_kwh" in fc.columns:
            self.assertTrue((fc["upper_kwh"] >= fc["predicted_kwh"]).all())
            self.assertTrue((fc["lower_kwh"] <= fc["predicted_kwh"]).all())

    def test_tc11c_future_observations_never_used_in_recursive_forecast(self):
        # Poison future actual data to verify strict leakage protection
        origin = self.hourly.index[-100]
        hist_clean = self.hourly.loc[:origin]
        fc_clean = recursive_forecast(self.model, hist_clean, horizon=12, meta=self.meta)

        poisoned = self.hourly.copy()
        poisoned.loc[origin + pd.Timedelta(hours=1):] = 999999.0
        fc_poisoned = recursive_forecast(self.model, poisoned.loc[:origin], horizon=12, meta=self.meta)

        self.assertTrue(
            np.allclose(fc_clean["predicted_kwh"], fc_poisoned["predicted_kwh"]),
            "Forecast was affected by future observations!",
        )

    def test_tc12_demand_tier_classification(self):
        p25 = 10.0
        p95 = 30.0

        tier_low, exp_low, alert_low = classify_demand(5.0, p25, p95)
        self.assertEqual(tier_low, "LOW")
        self.assertFalse(alert_low)

        tier_norm, exp_norm, alert_norm = classify_demand(20.0, p25, p95)
        self.assertEqual(tier_norm, "NORMAL")
        self.assertFalse(alert_norm)

        tier_high, exp_high, alert_high = classify_demand(35.0, p25, p95)
        self.assertEqual(tier_high, "HIGH")
        self.assertTrue(alert_high)
        self.assertIn("exceeds", exp_high.lower())

    def test_tc13_forecast_csv_export_schema(self):
        fc = recursive_forecast(self.model, self.hourly, horizon=6, meta=self.meta, client="MT_001")
        csv_str = format_export_csv(fc)
        import io
        df_exported = pd.read_csv(io.StringIO(csv_str))

        required_cols = [
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
        for col in required_cols:
            self.assertIn(col, df_exported.columns, f"Required export column '{col}' missing from CSV.")
        self.assertEqual(len(df_exported), 6)
        self.assertEqual(df_exported["unit"].iloc[0], "kWh")

    def test_tc14_sha256_integrity_tampering_blocked(self):
        tmp = Path(tempfile.mkdtemp())
        for fname in ("model.joblib", "metadata.json"):
            (tmp / fname).write_bytes((self.model_dir / fname).read_bytes())

        # Tamper with model weights
        with open(tmp / "model.joblib", "ab") as fh:
            fh.write(b"corruption_bytes")

        with self.assertRaises(ForecastError):
            load_artefacts(tmp)

    def test_tc14b_metadata_feature_mismatch_blocked(self):
        tmp = Path(tempfile.mkdtemp())
        for fname in ("model.joblib", "metadata.json"):
            (tmp / fname).write_bytes((self.model_dir / fname).read_bytes())

        # Tamper with metadata feature list
        meta_dict = json.loads((tmp / "metadata.json").read_text(encoding="utf-8"))
        meta_dict["feature_columns"] = ["random_col_1", "random_col_2"]
        (tmp / "metadata.json").write_text(json.dumps(meta_dict), encoding="utf-8")

        with self.assertRaises(ForecastError):
            load_artefacts(tmp)

    def test_tc15_missing_model_handling(self):
        empty_dir = Path(tempfile.mkdtemp())
        with self.assertRaises(ForecastError):
            load_artefacts(empty_dir)


    def test_tc16_forecast_demand_classification_integrity(self):
        fc = recursive_forecast(self.model, self.hourly, horizon=6, meta=self.meta)
        self.assertIn("alert_level", fc.columns)
        self.assertIn("alert_status", fc.columns)
        valid_tiers = {"LOW", "NORMAL", "HIGH"}
        for tier in fc["alert_level"]:
            self.assertIn(tier, valid_tiers)
        for status in fc["alert_status"]:
            self.assertIn(status, {"ALERT", "NO_ALERT"})


if __name__ == "__main__":
    unittest.main()

