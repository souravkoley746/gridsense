"""Automated tests for dataset loading, parsing, and schema validation (Phase 7.1)."""
from __future__ import annotations

import gzip
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd

from src.data_loader import (
    PORTFOLIO,
    DataError,
    detect_dialect,
    list_clients,
    load_raw,
    sha256_file,
)


def write_uci_file(path: Path, rows: list[tuple[str, str, str]], sep=";", decimal=","):
    header = f'"";"MT_001";"MT_002"' if sep == ";" else '"timestamp","MT_001","MT_002"'
    lines = [header]
    for ts, v1, v2 in rows:
        if decimal == ",":
            v1_s = str(v1).replace(".", ",")
            v2_s = str(v2).replace(".", ",")
        else:
            v1_s = str(v1)
            v2_s = str(v2)
        lines.append(f'"{ts}"{sep}"{v1_s}"{sep}"{v2_s}"')
    path.write_text("\n".join(lines), encoding="utf-8")


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def test_tc01_valid_csv_and_txt_loading(self):
        txt_path = self.tmp_dir / "valid.txt"
        write_uci_file(txt_path, [
            ("2012-01-01 00:15:00", "1.5", "2.0"),
            ("2012-01-01 00:30:00", "1.75", "2.25"),
        ])
        clients = list_clients(txt_path)
        self.assertEqual(clients, ["MT_001", "MT_002"])

        series, info = load_raw(txt_path, "MT_001")
        self.assertAlmostEqual(series.iloc[0], 1.5)
        self.assertAlmostEqual(series.iloc[1], 1.75)
        self.assertEqual(info["source_unit"], "kW")
        self.assertEqual(info["raw_rows"], 2)

    def test_tc01b_zip_loading_and_member_selection(self):
        txt_path = self.tmp_dir / "data.txt"
        write_uci_file(txt_path, [
            ("2012-01-01 00:15:00", "3.0", "4.0"),
            ("2012-01-01 00:30:00", "3.5", "4.5"),
        ])
        zip_path = self.tmp_dir / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(txt_path, arcname="inner_data.txt")

        clients = list_clients(zip_path)
        self.assertEqual(clients, ["MT_001", "MT_002"])
        series, info = load_raw(zip_path, "MT_002")
        self.assertAlmostEqual(series.iloc[0], 4.0)
        self.assertEqual(info["archive_member"], "inner_data.txt")

    def test_tc01c_gzip_loading(self):
        txt_path = self.tmp_dir / "data.csv"
        write_uci_file(txt_path, [
            ("2012-01-01 00:15:00", "5.0", "6.0"),
            ("2012-01-01 00:30:00", "5.5", "6.5"),
        ], sep=",", decimal=".")
        gz_path = self.tmp_dir / "dataset.csv.gz"
        with open(txt_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.writelines(f_in)

        clients = list_clients(gz_path)
        self.assertIn("MT_001", clients)
        series, info = load_raw(gz_path, "MT_001")
        self.assertAlmostEqual(series.iloc[0], 5.0)

    def test_tc01d_delimiter_detection(self):
        semi_path = self.tmp_dir / "semi.txt"
        comma_path = self.tmp_dir / "comma.csv"
        write_uci_file(semi_path, [("2012-01-01 00:15:00", "1.0", "2.0")], sep=";", decimal=",")
        write_uci_file(comma_path, [("2012-01-01 00:15:00", "1.0", "2.0")], sep=",", decimal=".")

        sep_s, dec_s = detect_dialect(semi_path)
        sep_c, dec_c = detect_dialect(comma_path)
        self.assertEqual(sep_s, ";")
        self.assertEqual(dec_s, ",")
        self.assertEqual(sep_c, ",")
        self.assertEqual(dec_c, ".")

    def test_tc02_invalid_schema_detected(self):
        bad_path = self.tmp_dir / "bad.txt"
        bad_path.write_text("only_one_column_here\n1\n2\n", encoding="utf-8")
        with self.assertRaises(DataError):
            list_clients(bad_path)
        with self.assertRaises(DataError):
            load_raw(bad_path, "MT_001")

    def test_tc03_duplicate_timestamps_rejected(self):
        dup_path = self.tmp_dir / "dup.txt"
        write_uci_file(dup_path, [
            ("2012-01-01 00:15:00", "1.0", "2.0"),
            ("2012-01-01 00:15:00", "1.5", "2.5"),
        ])
        with self.assertRaises(DataError):
            load_raw(dup_path, "MT_001")

    def test_tc04_non_numeric_values_rejected(self):
        nn_path = self.tmp_dir / "non_numeric.txt"
        write_uci_file(nn_path, [
            ("2012-01-01 00:15:00", "bad_text", "2.0"),
            ("2012-01-01 00:30:00", "1.5", "2.5"),
        ])
        with self.assertRaises(DataError):
            load_raw(nn_path, "MT_001")

    def test_tc04b_negative_values_counted(self):
        neg_path = self.tmp_dir / "neg.txt"
        write_uci_file(neg_path, [
            ("2012-01-01 00:15:00", "-5.0", "2.0"),
            ("2012-01-01 00:30:00", "3.0", "2.5"),
        ])
        series, info = load_raw(neg_path, "MT_001")
        self.assertEqual(info["negative_values"], 1)

    def test_tc04c_portfolio_aggregation(self):
        p_path = self.tmp_dir / "port.txt"
        write_uci_file(p_path, [
            ("2012-01-01 00:15:00", "2.0", "3.0"),
            ("2012-01-01 00:30:00", "4.0", "5.0"),
        ])
        series, info = load_raw(p_path, PORTFOLIO)
        self.assertTrue(info["is_portfolio"])
        self.assertAlmostEqual(series.iloc[0], 5.0)  # 2.0 + 3.0
        self.assertAlmostEqual(series.iloc[1], 9.0)  # 4.0 + 5.0


if __name__ == "__main__":
    unittest.main()
