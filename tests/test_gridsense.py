"""Automated tests mapped to the report's test cases TC01-TC15 (plus extras).
Run:  python -m unittest discover -s tests -v      (or: pytest)
"""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as config
from src.data_loader import DataError, list_clients, load_raw
from src.evaluate import alert_metrics, improvement_pct, point_metrics
from src.features import FEATURE_COLUMNS, InsufficientHistory, feature_row, make_supervised
from src.forecast import ForecastError, load_artefacts, recursive_forecast
from src.preprocessing import build_hourly


def write_uci(path: Path, rows: list[tuple[str, str, str]], header='"";"MT_001";"MT_002"'):
    path.write_text(header + "\n" + "\n".join(f'"{t}";"{a}";"{b}"' for t, a, b in rows), encoding="utf-8")


def kw_series(hours=60, value=4.0, start="2012-06-10 00:15:00"):
    idx = pd.date_range(start, periods=hours * 4, freq="15min")
    return pd.Series(value, index=idx, name="MT_001")


class DataLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_tc01_valid_file_parses(self):
        p = self.tmp / "ok.txt"
        write_uci(p, [("2012-06-10 00:15:00", "1,5", "2,0"), ("2012-06-10 00:30:00", "1,25", "2,5")])
        self.assertEqual(list_clients(p), ["MT_001", "MT_002"])
        s, info = load_raw(p, "MT_001")
        self.assertAlmostEqual(s.iloc[0], 1.5)       # decimal comma handled
        self.assertEqual(info["source_unit"], "kW")

    def test_tc02_invalid_schema(self):
        p = self.tmp / "bad.txt"
        p.write_text("only_one_column\n1\n2\n")
        with self.assertRaises(DataError):
            list_clients(p)
        with self.assertRaises(DataError):
            load_raw(self.tmp / "missing.txt", "MT_001")

    def test_tc03_duplicate_timestamps(self):
        p = self.tmp / "dup.txt"
        write_uci(p, [("2012-06-10 00:15:00", "1", "1"), ("2012-06-10 00:15:00", "2", "2")])
        with self.assertRaises(DataError):
            load_raw(p, "MT_001")

    def test_tc04_non_numeric(self):
        p = self.tmp / "nn.txt"
        write_uci(p, [("2012-06-10 00:15:00", "abc", "1"), ("2012-06-10 00:30:00", "2", "2")])
        with self.assertRaises(DataError):
            load_raw(p, "MT_001")

    def test_unknown_client(self):
        p = self.tmp / "ok.txt"
        write_uci(p, [("2012-06-10 00:15:00", "1", "1")])
        with self.assertRaises(DataError):
            load_raw(p, "NOPE")


class Preprocessing(unittest.TestCase):
    def test_tc05_energy_conversion(self):
        h, q = build_hourly(kw_series(60, 4.0))
        self.assertTrue(np.allclose(h["kwh"], 4.0))          # four readings of 4 kW = 4 kWh
        self.assertEqual(q["incomplete_hours"], 0)

    def test_tc06_incomplete_hour_flagged(self):
        s = kw_series(60, 4.0)
        s = s.drop(s.index[40])                              # one missing 15-min reading
        h, q = build_hourly(s)
        self.assertEqual(q["incomplete_hours"], 1)
        bad = h[h["incomplete"]]
        self.assertTrue(bad["imputed"].all())
        self.assertFalse(bad["scoreable"].any())              # never scored as ground truth

    def test_structural_zero_trim(self):
        s = kw_series(100, 4.0)
        s.iloc[:48] = 0.0
        h, q = build_hourly(s)
        self.assertEqual(q["structural_zero_rows_trimmed"], 48)

    def test_dst_hours_flagged(self):
        # 2012-03-25 is the last Sunday of March
        idx = pd.date_range("2012-03-24 00:15:00", periods=96 * 3, freq="15min")
        s = pd.Series(4.0, index=idx)
        s[(s.index > "2012-03-25 01:00") & (s.index <= "2012-03-25 02:00")] = 0.0
        h, q = build_hourly(s)
        self.assertEqual(q["dst_flagged_hours"], 1)
        self.assertFalse(h.loc["2012-03-25 02:00", "scoreable"])
        self.assertAlmostEqual(h.loc["2012-03-25 02:00", "kwh"], 4.0)   # causally repaired


class Features(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2012-01-01", periods=400, freq="h")
        self.y = pd.Series(np.arange(400, dtype=float), index=idx)

    def test_tc07_lag_alignment(self):
        f = make_supervised(self.y)
        r = f.iloc[10]
        pos = self.y.index.get_loc(f.index[10])
        self.assertEqual(r["lag_1"], self.y.iloc[pos - 1])
        self.assertEqual(r["lag_24"], self.y.iloc[pos - 24])
        self.assertEqual(r["lag_168"], self.y.iloc[pos - 168])

    def test_tc08_target_never_in_own_features(self):
        f1 = make_supervised(self.y)
        y2 = self.y.copy()
        t = f1.index[50]
        y2[t] += 1000.0                                       # change ONLY the target value
        f2 = make_supervised(y2)
        self.assertTrue(np.allclose(f1.loc[t, FEATURE_COLUMNS].astype(float), f2.loc[t, FEATURE_COLUMNS].astype(float)))

    def test_feature_row_matches_training_features(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.random(400) * 10, index=self.y.index)
        f = make_supervised(y)
        t = f.index[100]
        row = feature_row(y.loc[:t - pd.Timedelta(hours=1)], t)
        for c in FEATURE_COLUMNS:
            self.assertAlmostEqual(float(row[c]), float(f.loc[t, c]), places=9, msg=c)

    def test_incomplete_index_rejected(self):
        with self.assertRaises(ValueError):
            make_supervised(self.y.drop(self.y.index[5]))

    def test_tc10_insufficient_history(self):
        with self.assertRaises(InsufficientHistory):
            feature_row(self.y.iloc[:100], self.y.index[100])
        with self.assertRaises(InsufficientHistory):          # misaligned history end
            feature_row(self.y.iloc[:300], self.y.index[305])


class Metrics(unittest.TestCase):
    def test_point_metrics_and_sign(self):
        m = point_metrics([10, 10], [12, 14])
        self.assertAlmostEqual(m["MAE_kWh"], 3.0)
        self.assertGreater(m["ME_kWh"], 0)                    # overprediction is positive

    def test_improvement_zero_baseline_undefined(self):
        self.assertIsNone(improvement_pct(1.0, 0.0))
        self.assertAlmostEqual(improvement_pct(1.0, 2.0), 50.0)

    def test_tc13_alert_rule_and_confusion(self):
        a = alert_metrics([5, 12, 12, 1], [5, 12, 3, 11], threshold=10)
        self.assertEqual((a["TP"], a["FP"], a["FN"], a["TN"]), (1, 1, 1, 1))
        self.assertAlmostEqual(a["precision"], 0.5)

    def test_undefined_denominators(self):
        a = alert_metrics([1, 2], [1, 2], threshold=10)
        self.assertIsNone(a["precision"])
        self.assertIsNone(a["recall"])


class EndToEnd(unittest.TestCase):
    """Runs a reduced experiment once on the demo dataset inside a temp directory."""

    @classmethod
    def setUpClass(cls):
        from src import pipeline
        cls.tmp = Path(tempfile.mkdtemp())
        cls._orig_root = config.ROOT
        cfg = config.load_config()
        cfg = copy.deepcopy(cfg)
        cfg["dataset"]["path"] = str(cls._orig_root / "data" / "raw" / "LD2011_2014_demo.txt")
        cfg["models"]["random_forest"]["n_estimators"] = 20
        cfg["models"]["xgboost"]["n_estimators"] = 60
        cfg["uncertainty"]["backtest_origins"] = 6
        config.ROOT = cls.tmp
        cls.cfg = cfg
        cls.meta = pipeline.run_experiment("MT_003", cfg)
        cls.dir = config.model_dir("MT_003")

    @classmethod
    def tearDownClass(cls):
        config.ROOT = cls._orig_root

    def test_tc09_temporal_ordering(self):
        s = self.meta["splits"]
        self.assertLess(pd.Timestamp(s["train"][1]), pd.Timestamp(s["validation"][0]))
        self.assertLess(pd.Timestamp(s["validation"][1]), pd.Timestamp(s["test"][0]))

    def test_all_methods_scored_on_same_rows(self):
        n = {r["n"] for r in self.meta["test_metrics"]}
        self.assertEqual(len(n), 1)

    def test_tc11_model_loads_and_forecasts(self):
        model, meta = load_artefacts(self.dir)
        hourly = pd.read_csv(config.processed_path("MT_003"), index_col=0, parse_dates=True)["kwh"]
        fc = recursive_forecast(model, hourly, 24, meta)
        self.assertEqual(len(fc), 24)
        self.assertTrue(np.isfinite(fc["predicted_kwh"]).all())
        self.assertTrue((fc["upper_kwh"] >= fc["predicted_kwh"]).all())
        self.assertTrue((fc["lower_kwh"] <= fc["predicted_kwh"]).all())
        self.assertEqual(fc["target_time"].iloc[0], hourly.index[-1] + pd.Timedelta(hours=1))

    def test_forecast_never_uses_future_values(self):
        model, meta = load_artefacts(self.dir)
        hourly = pd.read_csv(config.processed_path("MT_003"), index_col=0, parse_dates=True)["kwh"]
        origin = hourly.index[-300]
        a = recursive_forecast(model, hourly.loc[:origin], 12, meta)
        poisoned = hourly.copy()
        poisoned.loc[origin + pd.Timedelta(hours=1):] = 1e6      # future corrupted
        b = recursive_forecast(model, poisoned.loc[:origin], 12, meta)
        self.assertTrue(np.allclose(a["predicted_kwh"], b["predicted_kwh"]))

    def test_tc12_metadata_mismatch_blocked(self):
        d = Path(tempfile.mkdtemp())
        for f in ("model.joblib", "metadata.json"):
            (d / f).write_bytes((self.dir / f).read_bytes())
        meta = json.loads((d / "metadata.json").read_text())
        meta["feature_columns"] = list(reversed(meta["feature_columns"]))
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(ForecastError):
            load_artefacts(d)

    def test_tampered_model_blocked(self):
        d = Path(tempfile.mkdtemp())
        for f in ("model.joblib", "metadata.json"):
            (d / f).write_bytes((self.dir / f).read_bytes())
        with open(d / "model.joblib", "ab") as fh:
            fh.write(b"x")
        with self.assertRaises(ForecastError):
            load_artefacts(d)

    def test_tc13_alert_flag_in_forecast(self):
        model, meta = load_artefacts(self.dir)
        hourly = pd.read_csv(config.processed_path("MT_003"), index_col=0, parse_dates=True)["kwh"]
        fc = recursive_forecast(model, hourly, 6, meta, threshold=0.0)
        self.assertTrue(fc["alert"].all())

    def test_tc14_export_csv_roundtrip(self):
        model, meta = load_artefacts(self.dir)
        hourly = pd.read_csv(config.processed_path("MT_003"), index_col=0, parse_dates=True)["kwh"]
        fc = recursive_forecast(model, hourly, 5, meta)
        p = Path(tempfile.mkdtemp()) / "f.csv"
        fc.to_csv(p, index=False)
        back = pd.read_csv(p, parse_dates=["issue_time", "target_time"])
        for c in ("issue_time", "target_time", "predicted_kwh", "model"):
            self.assertIn(c, back.columns)
        self.assertEqual(len(back), 5)

    def test_tc10_forecast_insufficient_history(self):
        model, meta = load_artefacts(self.dir)
        hourly = pd.read_csv(config.processed_path("MT_003"), index_col=0, parse_dates=True)["kwh"]
        with self.assertRaises(ForecastError):
            recursive_forecast(model, hourly.iloc[:50], 3, meta)


class MissingArtefact(unittest.TestCase):
    def test_tc15_missing_artefact(self):
        with self.assertRaises(ForecastError):
            load_artefacts(Path(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
