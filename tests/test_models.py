"""Automated tests for machine learning models, baselines, and chronological evaluation (Phase 7.4)."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

import src.config as config
from src.baselines import baseline_predictions
from src.evaluate import improvement_pct, point_metrics
from src.features import FEATURE_COLUMNS, make_supervised
from src.models import build_candidates
from src import pipeline


class TestModelsAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls._orig_root = config.ROOT
        config.ROOT = cls.tmp_dir

        cfg = config.load_config(cls._orig_root / "config" / "experiment.json")
        cfg = copy.deepcopy(cfg)
        cfg["dataset"]["path"] = str(cls._orig_root / "data" / "raw" / "LD2011_2014_demo.txt")
        cfg["models"]["random_forest"]["n_estimators"] = 15
        cfg["models"]["xgboost"]["n_estimators"] = 30
        cfg["uncertainty"]["backtest_origins"] = 5
        cls.cfg = cfg
        cls.meta = pipeline.run_experiment("MT_002", cfg)
        cls.model_dir = config.model_dir("MT_002")

    @classmethod
    def tearDownClass(cls):
        config.ROOT = cls._orig_root

    def test_tc09_chronological_data_split(self):
        splits = self.meta["splits"]
        train_end = pd.Timestamp(splits["train"][1])
        val_start = pd.Timestamp(splits["validation"][0])
        val_end = pd.Timestamp(splits["validation"][1])
        test_start = pd.Timestamp(splits["test"][0])

        # Strict chronological ordering: train < validation < test
        self.assertLess(train_end, val_start, "Train period must strictly precede validation period")
        self.assertLess(val_end, test_start, "Validation period must strictly precede test period")

    def test_tc09b_all_five_models_scored_on_same_test_timestamps(self):
        metrics = self.meta["test_metrics"]
        model_names = [m["model_name"] for m in metrics]

        # All 5 models must be present
        expected_models = [
            "Persistence",
            "Daily seasonal-naive",
            "Weekly seasonal-naive",
            "Random Forest",
            "XGBoost",
        ]
        for em in expected_models:
            self.assertIn(em, model_names, f"Expected model '{em}' missing from test metrics.")

        # Scored on the exact same test hours
        counts = {m["sample_count"] for m in metrics}
        self.assertEqual(len(counts), 1, "All models must be evaluated on the identical test sample count.")

    def test_tc09c_point_metrics_and_bias_definition(self):
        # Actual: [10, 10], Predicted: [12, 14] -> errors: [+2, +4]
        # MAE = 3.0, RMSE = sqrt((4 + 16)/2) = sqrt(10) = 3.162, ME = +3.0
        m = point_metrics([10.0, 10.0], [12.0, 14.0])
        self.assertAlmostEqual(m["MAE_kWh"], 3.0)
        self.assertAlmostEqual(m["RMSE_kWh"], np.sqrt(10.0))
        self.assertAlmostEqual(m["ME_kWh"], 3.0)

        # Underprediction: actual: [20], pred: [15] -> diff = -5
        m_under = point_metrics([20.0], [15.0])
        self.assertLess(m_under["ME_kWh"], 0, "Underprediction must yield negative ME")

    def test_tc09d_relative_improvement_calculation(self):
        # Baseline = 2.0 kWh, Model = 1.0 kWh -> 50% improvement
        imp = improvement_pct(model_error=1.0, baseline_error=2.0)
        self.assertAlmostEqual(imp, 50.0)

        # Worse than baseline: Baseline = 2.0, Model = 3.0 -> -50% improvement
        imp_neg = improvement_pct(model_error=3.0, baseline_error=2.0)
        self.assertAlmostEqual(imp_neg, -50.0)

    def test_tc09e_baseline_models_predictions(self):
        idx = pd.date_range("2012-01-01 00:00:00", periods=200, freq="h")
        y = pd.Series(np.arange(200, dtype=float), index=idx)
        df = make_supervised(y)
        preds = baseline_predictions(df)

        self.assertIn("Persistence", preds.columns)
        self.assertIn("Daily seasonal-naive", preds.columns)
        self.assertIn("Weekly seasonal-naive", preds.columns)
        self.assertTrue(np.allclose(preds["Persistence"], df["lag_1"]))
        self.assertTrue(np.allclose(preds["Daily seasonal-naive"], df["lag_24"]))
        self.assertTrue(np.allclose(preds["Weekly seasonal-naive"], df["lag_168"]))


if __name__ == "__main__":
    unittest.main()
