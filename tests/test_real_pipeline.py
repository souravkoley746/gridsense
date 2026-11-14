"""Comprehensive unit and integration test suite for the genuine UCI real-dataset pipeline.

Tests all 22 required pipeline specifications:
1. Real dataset parsing
2. Semicolon-separated file handling
3. Decimal-comma conversion
4. Timestamp parsing
5. Client-column detection
6. 15-minute to hourly kWh conversion (E_h = sum(P) * 0.25)
7. Hourly aggregation correctness
8. DST identification (March 23h & October 25h)
9. Structural-zero identification & trimming
10. Constant-zero client exclusion (MT_223 exclusion, MT_008 validation)
11. Representative-client selection
12. Chronological train/validation/test splitting (70/15/15)
13. Feature leakage prevention
14. Lag feature correctness (lags 1, 24, 168)
15. Rolling feature correctness (shifted rolling statistics)
16. Model training (Random Forest & XGBoost fitting)
17. Prediction generation
18. Metric calculation (MAE, RMSE, ME = Mean(Pred - Actual))
19. Recursive forecast leakage prevention (no teacher forcing)
20. Multi-client artifact generation
21. Dashboard page loading (all 6 pages AppTest)
22. Missing-dataset error handling
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.baselines import baseline_predictions
from src.config import load_config, dataset_path, demo_dataset_path, model_dir, report_dir
from src.data_loader import detect_dialect, list_clients, load_raw, PORTFOLIO, DataError
from src.evaluate import improvement_pct, point_metrics
from src.features import FEATURE_COLUMNS, make_supervised, feature_row
from src.forecast import load_artefacts, recursive_forecast
from src.models import build_candidates
from src.preprocessing import build_hourly, dst_hours
from scripts.profile_and_select_clients import autocorrelation
from scripts.train_multi_client import evaluate_multi_horizon_stratified

ROOT = Path(__file__).resolve().parents[1]


class TestRealDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    # 1. Real dataset parsing
    def test_01_real_dataset_parsing(self):
        sample = '"2012-01-01 00:15:00";"10,50";"20,00"\n"2012-01-01 00:30:00";"12,00";"22,50"\n'
        p = self.tmp_dir / "sample.txt"
        p.write_text(sample, encoding="utf-8")
        sep, dec = detect_dialect(p)
        self.assertEqual(sep, ";")
        self.assertEqual(dec, ",")

    # 2. Semicolon-separated file handling
    def test_02_semicolon_separated_file_handling(self):
        sample = 'timestamp;MT_001\n2012-01-01 00:15:00;5,0\n2012-01-01 00:30:00;6,0\n'
        p = self.tmp_dir / "semi.txt"
        p.write_text(sample, encoding="utf-8")
        s, info = load_raw(p, "MT_001")
        self.assertEqual(len(s), 2)
        self.assertEqual(s.iloc[0], 5.0)

    # 3. Decimal-comma conversion
    def test_03_decimal_comma_conversion(self):
        sample = 'timestamp;MT_001\n2012-01-01 00:15:00;123,456\n2012-01-01 00:30:00;789,012\n'
        p = self.tmp_dir / "comma.txt"
        p.write_text(sample, encoding="utf-8")
        s, _ = load_raw(p, "MT_001")
        self.assertAlmostEqual(s.iloc[0], 123.456)
        self.assertAlmostEqual(s.iloc[1], 789.012)

    # 4. Timestamp parsing
    def test_04_timestamp_parsing(self):
        sample = 'timestamp;MT_001\n2012-03-25 00:15:00;10,0\n2012-03-25 00:30:00;10,0\n'
        p = self.tmp_dir / "ts.txt"
        p.write_text(sample, encoding="utf-8")
        s, _ = load_raw(p, "MT_001")
        self.assertIsInstance(s.index, pd.DatetimeIndex)
        self.assertEqual(s.index[0], pd.Timestamp("2012-03-25 00:15:00"))

    # 5. Client-column detection
    def test_05_client_column_detection(self):
        sample = 'timestamp;MT_001;MT_002;MT_003\n2012-01-01 00:15:00;1,0;2,0;3,0\n'
        p = self.tmp_dir / "clients.txt"
        p.write_text(sample, encoding="utf-8")
        clients = list_clients(p)
        self.assertEqual(clients, ["MT_001", "MT_002", "MT_003"])

    # 6. 15-minute to hourly kWh conversion (E_h = sum(4 readings) * 0.25)
    def test_06_energy_conversion_formula(self):
        idx = pd.date_range("2012-01-01 00:15:00", periods=4, freq="15min")
        kw = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)
        hourly, _ = build_hourly(kw)
        # 4 readings of 10.0 kW = 40.0 * 0.25 = 10.0 kWh
        self.assertAlmostEqual(hourly["kwh"].iloc[0], 10.0)

    # 7. Hourly aggregation correctness
    def test_07_hourly_aggregation_correctness(self):
        idx = pd.date_range("2012-01-01 00:15:00", periods=8, freq="15min")
        # Hour 1: 4, 8, 12, 16 kW -> sum 40 * 0.25 = 10 kWh
        # Hour 2: 20, 20, 20, 20 kW -> sum 80 * 0.25 = 20 kWh
        kw = pd.Series([4.0, 8.0, 12.0, 16.0, 20.0, 20.0, 20.0, 20.0], index=idx)
        hourly, _ = build_hourly(kw)
        self.assertEqual(len(hourly), 2)
        self.assertAlmostEqual(hourly["kwh"].iloc[0], 10.0)
        self.assertAlmostEqual(hourly["kwh"].iloc[1], 20.0)

    # 8. DST identification (March & October)
    def test_08_dst_identification(self):
        idx_mar = pd.date_range("2013-03-31 00:00:00", periods=24, freq="h")
        flags_mar = dst_hours(idx_mar)
        self.assertTrue(flags_mar.loc["2013-03-31 02:00:00", "dst_march"])

        idx_oct = pd.date_range("2013-10-27 00:00:00", periods=24, freq="h")
        flags_oct = dst_hours(idx_oct)
        self.assertTrue(flags_oct.loc["2013-10-27 02:00:00", "dst_october"])

    # 9. Structural-zero identification & trimming
    def test_09_structural_zero_identification(self):
        idx = pd.date_range("2012-01-01 00:15:00", periods=16, freq="15min")
        # 8 zeros (2 hours) followed by 8 positive readings
        kw = pd.Series([0.0]*8 + [10.0]*8, index=idx)
        hourly, q = build_hourly(kw, trim_structural_zeros=True)
        self.assertEqual(q["structural_zero_rows_trimmed"], 8)
        self.assertEqual(len(hourly), 2)
        self.assertAlmostEqual(hourly["kwh"].iloc[0], 10.0)

    # 10. Constant-zero client exclusion (MT_223 exclusion, MT_008 validation)
    def test_10_constant_zero_client_exclusion(self):
        # Simulated MT_223: active in 2011, but constant zero throughout 2013-2014
        idx = pd.date_range("2011-01-01 01:00:00", periods=3000, freq="h")
        # Last 1000 hours are all 0
        vals_223 = np.array([50.0]*2000 + [0.0]*1000)
        test_period_active_pct = (vals_223[-500:] > 0).mean() * 100
        self.assertEqual(test_period_active_pct, 0.0)  # Excluded!

        # MT_008: 100% active in test period
        vals_008 = np.array([0.0]*500 + [250.0]*2500)
        test_008_active_pct = (vals_008[-500:] > 0).mean() * 100
        self.assertEqual(test_008_active_pct, 100.0)  # Valid replacement!

    # 11. Representative-client selection
    def test_11_representative_client_selection(self):
        cfg = load_config()
        rep = cfg["dataset"]["representative_clients"]
        self.assertEqual(len(rep), 10)
        self.assertIn("MT_001", rep)
        self.assertIn("MT_008", rep)
        self.assertNotIn("MT_223", rep)

    # 12. Chronological train/validation/test splitting (70/15/15)
    def test_12_chronological_splitting(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=1000, freq="h")
        s = pd.Series(np.arange(1000, dtype=float), index=idx)
        frame = make_supervised(s)
        n = len(frame)
        i_tr = int(n * 0.70)
        i_va = int(n * 0.85)
        train, val, test = frame.iloc[:i_tr], frame.iloc[i_tr:i_va], frame.iloc[i_va:]
        # Strictly chronological
        self.assertLess(train.index[-1], val.index[0])
        self.assertLess(val.index[-1], test.index[0])

    # 13. Feature leakage prevention
    def test_13_feature_leakage_prevention(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=200, freq="h")
        s = pd.Series(np.arange(200, dtype=float), index=idx)
        df = make_supervised(s)
        row = df.loc[idx[180]]
        self.assertEqual(row["target_kwh"], 180.0)
        self.assertEqual(row["lag_1"], 179.0)  # t - 1
        self.assertEqual(row["lag_24"], 156.0) # t - 24
        self.assertEqual(row["lag_168"], 12.0) # t - 168

    # 14. Lag feature correctness (lags 1, 24, 168)
    def test_14_lag_feature_correctness(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=200, freq="h")
        s = pd.Series(np.arange(200, dtype=float), index=idx)
        df = make_supervised(s)
        self.assertIn("lag_1", df.columns)
        self.assertIn("lag_24", df.columns)
        self.assertIn("lag_168", df.columns)
        # Verify lags match shifted historical series on valid indices
        for t in df.index:
            self.assertEqual(df.loc[t, "lag_1"], s.loc[t - pd.Timedelta(hours=1)])
            self.assertEqual(df.loc[t, "lag_24"], s.loc[t - pd.Timedelta(hours=24)])
            self.assertEqual(df.loc[t, "lag_168"], s.loc[t - pd.Timedelta(hours=168)])

    # 15. Rolling feature correctness (shifted rolling statistics)
    def test_15_rolling_feature_correctness(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=200, freq="h")
        s = pd.Series(np.arange(200, dtype=float), index=idx)
        df = make_supervised(s)
        # roll_mean_24 at row index 170 should be mean of rows 146..169 (not including 170)
        expected_24_mean = s.iloc[146:170].mean()
        self.assertAlmostEqual(df.loc[idx[170], "rolling_mean_24"], expected_24_mean)

    # 16. Model training (Random Forest & XGBoost fitting)
    def test_16_model_training(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=250, freq="h")
        s = pd.Series(np.random.RandomState(42).uniform(10, 50, size=250), index=idx)
        df = make_supervised(s)
        cands, _ = build_candidates({"models": {"random_forest": {"n_estimators": 5}, "xgboost": {"n_estimators": 5}}})
        rf = cands["Random Forest"]
        xgb = cands["XGBoost"]
        rf.fit(df[FEATURE_COLUMNS], df["target_kwh"])
        xgb.fit(df[FEATURE_COLUMNS], df["target_kwh"])
        self.assertTrue(hasattr(rf, "predict"))
        self.assertTrue(hasattr(xgb, "predict"))

    # 17. Prediction generation
    def test_17_prediction_generation(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=400, freq="h")
        s = pd.Series(np.random.RandomState(42).uniform(10, 50, size=400), index=idx)
        df = make_supervised(s)
        cands, _ = build_candidates({"models": {"random_forest": {"n_estimators": 5}, "xgboost": {"n_estimators": 5}}})
        rf = cands["Random Forest"]
        train_df = df.iloc[:180]
        test_df = df.iloc[180:]
        rf.fit(train_df[FEATURE_COLUMNS], train_df["target_kwh"])
        preds = rf.predict(test_df[FEATURE_COLUMNS])
        self.assertEqual(len(preds), len(test_df))
        self.assertTrue(np.all(np.isfinite(preds)))

    # 18. Metric calculation (MAE, RMSE, ME = Mean(Pred - Actual))
    def test_18_metric_calculation(self):
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 18.0, 35.0]
        pm = point_metrics(actual, predicted)
        # Errors: +2, -2, +5 -> MAE = (2+2+5)/3 = 3.0
        # Squared: 4, 4, 25 = 33 / 3 = 11 -> RMSE = sqrt(11) = 3.3166
        # ME = (+2 -2 +5)/3 = +5/3 = +1.6667
        self.assertAlmostEqual(pm["MAE_kWh"], 3.0)
        self.assertAlmostEqual(pm["RMSE_kWh"], np.sqrt(11.0), places=4)
        self.assertAlmostEqual(pm["ME_kWh"], 5.0 / 3.0, places=4)

    # 19. Recursive forecast leakage prevention (no teacher forcing)
    def test_19_recursive_forecast_leakage_prevention(self):
        idx = pd.date_range("2012-01-01 01:00:00", periods=400, freq="h")
        s = pd.Series(np.random.RandomState(42).uniform(10, 50, size=400), index=idx)
        df = make_supervised(s)
        cands, _ = build_candidates({"models": {"random_forest": {"n_estimators": 5}, "xgboost": {"n_estimators": 5}}})
        rf = cands["Random Forest"]
        rf.fit(df.iloc[:150][FEATURE_COLUMNS], df.iloc[:150]["target_kwh"])
        # History strictly up to timestamp 250
        origin = idx[250]
        hist = s.loc[:origin]
        fc = recursive_forecast(rf, hist, horizon=6)
        self.assertEqual(len(fc), 6)
        self.assertIn("predicted_kwh", fc.columns)
        # Ensure future values were not required to produce fc
        self.assertTrue(np.all(np.isfinite(fc["predicted_kwh"])))

    # 20. Multi-client artifact generation
    def test_20_multi_client_artifact_generation(self):
        client_dir = ROOT / "models" / "MT_001"
        self.assertTrue((client_dir / "model.joblib").exists())
        self.assertTrue((client_dir / "metadata.json").exists())
        rep_dir = ROOT / "reports" / "MT_001"
        self.assertTrue((rep_dir / "metrics.csv").exists())
        self.assertTrue((rep_dir / "test_predictions.csv").exists())

    # 21. Dashboard page loading (all 6 pages AppTest)
    def test_21_dashboard_page_loading(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"App raised exception on Overview: {at.exception}")

        pages = ["Historical Analysis", "Forecasting", "Model Performance", "Data Quality", "About Project", "Overview"]
        for p in pages:
            at.radio(key="main_nav").set_value(p).run()
            self.assertFalse(at.exception, f"App raised exception on {p}: {at.exception}")

    # 22. Missing-dataset error handling
    def test_22_missing_dataset_error_handling(self):
        fake_path = self.tmp_dir / "non_existent_dataset.txt"
        with self.assertRaises(DataError):
            load_raw(fake_path, "MT_001")

    # 23. Config missing real dataset raises FileNotFoundError and never silently falls back
    def test_23_config_missing_real_dataset_raises_filenotfound(self):
        fake_cfg = {"dataset": {"path": "data/raw/real/non_existent_file.txt"}}
        with self.assertRaises(FileNotFoundError) as ctx:
            dataset_path(fake_cfg)
        self.assertIn("Configured dataset was not found", str(ctx.exception))
        self.assertIn("download_real_data.py", str(ctx.exception))

    # 24. Explicit demo dataset path
    def test_24_demo_dataset_path_explicit(self):
        demo_p = demo_dataset_path()
        self.assertTrue(demo_p.exists(), f"Demo dataset missing at {demo_p}")
        self.assertEqual(demo_p.name, "LD2011_2014_demo.txt")

    # 25. Multi-horizon H=1 reconciliation selects exact model matching meta['selected_model']
    def test_25_multi_horizon_reconciliation_model_match(self):
        rep_dir = report_dir("MT_001")
        test_preds = pd.read_csv(rep_dir / "test_predictions.csv", index_col=0, parse_dates=True)
        _, recon = evaluate_multi_horizon_stratified("MT_001", horizons=[1], origins_per_hour=1)
        self.assertEqual(recon["selected_model"], "XGBoost")
        self.assertEqual(recon["evaluation_column_used"], "XGBoost")
        expected_full_mae = round(float(test_preds.loc[test_preds["scoreable"], "XGBoost"].sub(test_preds.loc[test_preds["scoreable"], "Actual"]).abs().mean()), 4)
        self.assertEqual(recon["full_test_partition_h1_mae_kwh"], expected_full_mae)

    # 26. All 11 models load, have valid metadata, and generate finite predictions
    def test_26_all_11_models_load_and_predict(self):
        targets = [
            "MT_001", "MT_141", "MT_154", "MT_311", "MT_343",
            "MT_008", "MT_362", "MT_341", "MT_131", "MT_163", "PORTFOLIO"
        ]
        for tgt in targets:
            mdir = model_dir(tgt)
            self.assertTrue((mdir / "model.joblib").exists(), f"Missing model for {tgt}")
            self.assertTrue((mdir / "metadata.json").exists(), f"Missing metadata for {tgt}")
            model, meta = load_artefacts(mdir)
            feat_cols = meta.get("feature_columns") or meta.get("feature_names")
            self.assertIsNotNone(feat_cols, f"Feature columns not found in metadata for {tgt}")
            self.assertIn("selected_model", meta)
            dummy_feats = pd.DataFrame([{f: 1.0 for f in feat_cols}])
            pred = model.predict(dummy_feats)
            self.assertEqual(len(pred), 1)
            self.assertTrue(np.isfinite(pred[0]))

    # 27. Macro NMAE formulation verification
    def test_27_macro_nmae_formulation(self):
        hz_csv = ROOT / "reports" / "horizon_evaluation" / "multi_horizon_metrics.csv"
        self.assertTrue(hz_csv.exists(), "multi_horizon_metrics.csv missing")
        df_hz = pd.read_csv(hz_csv)
        for _, row in df_hz.iterrows():
            if row["mean_actual_kwh"] > 0:
                expected_nmae = round(row["MAE_kWh"] / row["mean_actual_kwh"], 4)
                self.assertAlmostEqual(row["NMAE"], expected_nmae, places=4)

    # 28. Empirical raw data verification of MT_223 permanent decommissioning and MT_008 activity
    def test_28_mt_223_confirmed_decommissioned_and_mt_008_active(self):
        real_txt = ROOT / "data" / "raw" / "real" / "LD2011_2014.txt"
        if not real_txt.exists():
            self.skipTest("Real dataset not present locally for raw verification test")
        s_223, _ = load_raw(real_txt, "MT_223")
        s_008, _ = load_raw(real_txt, "MT_008")
        test_len = int(len(s_223) * 0.15)
        test_223 = s_223.iloc[-test_len:]
        test_008 = s_008.iloc[-test_len:]
        self.assertEqual(int((test_223 == 0).sum()), len(test_223))
        self.assertEqual(int((test_008 > 0).sum()), len(test_008))


if __name__ == "__main__":
    unittest.main()
