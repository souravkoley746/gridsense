"""Automated tests for data preprocessing, kWh conversion, and DST handling (Phase 7.2)."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.preprocessing import build_hourly, dst_hours


def make_kw_series(hours=48, value=4.0, start="2012-06-10 00:15:00"):
    idx = pd.date_range(start, periods=hours * 4, freq="15min")
    return pd.Series(value, index=idx, name="MT_001")


class TestPreprocessing(unittest.TestCase):
    def test_tc05_hourly_energy_conversion(self):
        # 4 readings of 4.0 kW in an hour = 4.0 * 4 * 0.25 = 4.0 kWh
        kw = make_kw_series(hours=24, value=4.0)
        hourly, q = build_hourly(kw)
        self.assertTrue(np.allclose(hourly["kwh"], 4.0))
        self.assertEqual(q["incomplete_hours"], 0)
        self.assertEqual(q["imputed_hours"], 0)

    def test_tc05b_varying_readings_conversion(self):
        # P1=2.0, P2=4.0, P3=6.0, P4=8.0 -> sum = 20.0 -> * 0.25 = 5.0 kWh
        idx = pd.date_range("2012-06-10 00:15:00", periods=4, freq="15min")
        kw = pd.Series([2.0, 4.0, 6.0, 8.0], index=idx, name="MT_001")
        hourly, q = build_hourly(kw)
        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(hourly["kwh"].iloc[0], 5.0)

    def test_tc06_incomplete_hour_flagged_and_excluded_from_scoring(self):
        kw = make_kw_series(hours=48, value=4.0)
        # Drop one 15-minute observation in hour 10
        kw = kw.drop(kw.index[41])
        hourly, q = build_hourly(kw)
        self.assertEqual(q["incomplete_hours"], 1)

        # Incomplete hour must be flagged and not scoreable
        inc_rows = hourly[hourly["incomplete"]]
        self.assertEqual(len(inc_rows), 1)
        self.assertFalse(inc_rows["scoreable"].any())

    def test_tc06b_structural_zero_trimming(self):
        kw = make_kw_series(hours=60, value=5.0)
        # First 40 readings are structural zeros before client joins grid
        kw.iloc[:40] = 0.0
        hourly, q = build_hourly(kw, trim_structural_zeros=True)
        self.assertEqual(q["structural_zero_rows_trimmed"], 40)
        # Analysis should start at first positive reading
        self.assertEqual(q["analysis_start_after_trim"], str(kw.index[40]))

    def test_tc06c_legitimate_internal_zeros_preserved(self):
        kw = make_kw_series(hours=48, value=5.0)
        # Internal zeros (e.g., zero consumption during maintenance or night)
        kw.iloc[20:24] = 0.0
        hourly, q = build_hourly(kw, trim_structural_zeros=True)
        # Internal zero readings should not be trimmed as structural prefix
        self.assertEqual(q["structural_zero_rows_trimmed"], 0)
        zero_hours = hourly[hourly["kwh"] == 0.0]
        self.assertGreater(len(zero_hours), 0)

    def test_tc06d_dst_flagging_and_exclusion(self):
        # 2012-03-25 is the last Sunday of March in European DST
        idx = pd.date_range("2012-03-24 00:15:00", periods=96 * 3, freq="15min")
        kw = pd.Series(4.0, index=idx)
        # March DST 01:00-02:00 has zeros in UCI dataset
        kw[(kw.index > "2012-03-25 01:00") & (kw.index <= "2012-03-25 02:00")] = 0.0

        hourly, q = build_hourly(kw, dst_policy="flag_only")
        self.assertGreater(q["dst_flagged_hours"], 0)
        dst_row = hourly.loc["2012-03-25 02:00"]
        self.assertTrue(dst_row["dst"])
        # DST affected hours must never be scored as ground truth target
        self.assertFalse(dst_row["scoreable"])


    def test_tc06e_pandas_monthly_resampling_compatibility(self):
        """Verify that month-end resampling succeeds under both Pandas 2.x and Pandas 3.x."""
        idx = pd.date_range("2012-01-01 01:00:00", periods=24 * 90, freq="h")
        s = pd.Series(10.0, index=idx)
        try:
            resampled = s.resample("ME").sum()
        except ValueError:
            resampled = s.resample("M").sum()
        self.assertGreater(len(resampled), 0)
        self.assertTrue((resampled > 0).all())


if __name__ == "__main__":
    unittest.main()

