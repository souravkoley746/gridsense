"""Automated tests for feature engineering, leakage prevention, and real-time alignment (Phase 7.3)."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    InsufficientHistory,
    feature_row,
    make_supervised,
)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2012-01-01 00:00:00", periods=400, freq="h")
        self.y = pd.Series(np.arange(400, dtype=float), index=idx)

    def test_tc07_day_of_month_included(self):
        self.assertIn("day_of_month", FEATURE_COLUMNS)
        df = make_supervised(self.y)
        self.assertIn("day_of_month", df.columns)
        for ts, row in df.iloc[:10].iterrows():
            self.assertEqual(row["day_of_month"], ts.day)

    def test_tc07b_lag_alignments(self):
        df = make_supervised(self.y)
        test_row = df.iloc[20]
        pos = self.y.index.get_loc(df.index[20])
        # lag_1 is y at pos - 1
        self.assertEqual(test_row["lag_1"], self.y.iloc[pos - 1])
        # lag_24 is y at pos - 24
        self.assertEqual(test_row["lag_24"], self.y.iloc[pos - 24])
        # lag_168 is y at pos - 168
        self.assertEqual(test_row["lag_168"], self.y.iloc[pos - 168])

    def test_tc08_target_never_leaks_into_own_features(self):
        f1 = make_supervised(self.y)
        y_mutated = self.y.copy()
        target_timestamp = f1.index[30]
        # Mutate ONLY the target value at this timestamp by an extreme amount
        y_mutated[target_timestamp] += 999999.0
        f2 = make_supervised(y_mutated)

        # The features constructed for target_timestamp MUST NOT change at all!
        f1_features = f1.loc[target_timestamp, FEATURE_COLUMNS].to_numpy(dtype=float)
        f2_features = f2.loc[target_timestamp, FEATURE_COLUMNS].to_numpy(dtype=float)
        self.assertTrue(
            np.allclose(f1_features, f2_features),
            "Target observation leaked into its own feature vector!",
        )

    def test_tc08b_rolling_window_calculation(self):
        df = make_supervised(self.y)
        t = df.index[50]
        pos = self.y.index.get_loc(t)
        # 24h rolling mean of observations strictly before t: [pos-24 : pos]
        expected_rolling_mean_24 = self.y.iloc[pos - 24: pos].mean()
        self.assertAlmostEqual(df.loc[t, "rolling_mean_24"], expected_rolling_mean_24)

    def test_tc08c_single_row_feature_generation_alignment(self):
        rng = np.random.default_rng(42)
        random_y = pd.Series(rng.uniform(5, 50, size=len(self.y)), index=self.y.index)
        df = make_supervised(random_y)
        t = df.index[80]

        # feature_row takes history strictly up to t - 1h
        history = random_y.loc[: t - pd.Timedelta(hours=1)]
        row = feature_row(history, t)

        for col in FEATURE_COLUMNS:
            self.assertAlmostEqual(
                float(row[col]),
                float(df.loc[t, col]),
                places=7,
                msg=f"Feature row mismatch for feature '{col}'",
            )

    def test_tc10_insufficient_history_raises_exception(self):
        # Requires >= 168 hours of history
        with self.assertRaises(InsufficientHistory):
            feature_row(self.y.iloc[:100], self.y.index[100])

    def test_tc10b_misaligned_history_raises_exception(self):
        # History ends at t - 2h instead of t - 1h
        t = self.y.index[200]
        history = self.y.loc[: t - pd.Timedelta(hours=2)]
        with self.assertRaises(InsufficientHistory):
            feature_row(history, t)


if __name__ == "__main__":
    unittest.main()
