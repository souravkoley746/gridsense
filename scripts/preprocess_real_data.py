"""Preprocess genuine UCI electricity consumption dataset from 15-minute kW into hourly kWh.

Core Transformations:
1. Physical Energy Conversion: E_h = sum(4 readings) * 0.25 kWh.
2. Incomplete Hour Exclusion: Hours with < 4 readings marked incomplete and excluded from scoring.
3. European Daylight Saving Time (DST) Handling: March & October transition hours flagged.
4. Structural Leading Zero Trimming: Trims pre-activation zeros on a per-client basis.
5. Quality Audit: Generates preprocessing quality metrics and writes to reports/data_quality/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL_TXT = ROOT / "data" / "raw" / "real" / "LD2011_2014.txt"
PROCESSED_DIR = ROOT / "data" / "processed"
QUALITY_DIR = ROOT / "reports" / "data_quality"

SELECTED_10_CLIENTS = [
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


def get_last_sunday(year: int, month: int) -> pd.Timestamp:
    last_day = 31
    d = pd.Timestamp(year=year, month=month, day=last_day)
    offset = (d.dayofweek + 1) % 7
    return (d - pd.Timedelta(days=offset)).normalize()


def dst_flag_series(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    df_flags = pd.DataFrame(index=hourly_index)
    df_flags["dst_march"] = False
    df_flags["dst_october"] = False

    years = hourly_index.year.unique()
    for yr in years:
        mar_sun = get_last_sunday(yr, 3)
        mar_ts = mar_sun + pd.Timedelta(hours=2)
        if mar_ts in df_flags.index:
            df_flags.loc[mar_ts, "dst_march"] = True

        oct_sun = get_last_sunday(yr, 10)
        oct_ts = oct_sun + pd.Timedelta(hours=2)
        if oct_ts in df_flags.index:
            df_flags.loc[oct_ts, "dst_october"] = True

    df_flags["dst"] = df_flags["dst_march"] | df_flags["dst_october"]
    return df_flags


def build_hourly_kwh(kw_series: pd.Series, trim_leading_zeros: bool = True) -> tuple[pd.DataFrame, dict]:
    """Convert 15-minute kW series to hourly kWh energy series with quality flags."""
    raw_len = len(kw_series)
    start_raw = kw_series.index.min()
    end_raw = kw_series.index.max()

    # Leading structural zeros
    leading_zeros = 0
    if trim_leading_zeros:
        non_zeros = kw_series[kw_series > 0]
        if len(non_zeros) > 0:
            first_active_ts = non_zeros.index[0]
            leading_zeros = int((kw_series.index < first_active_ts).sum())
            kw_active = kw_series.loc[first_active_ts:]
        else:
            kw_active = kw_series
            leading_zeros = raw_len
    else:
        kw_active = kw_series

    # Exact physical conversion: E_h = sum(P_j) * 0.25 kWh
    hourly_sum_kw = kw_active.resample("1h").sum()
    hourly_count = kw_active.resample("1h").count()
    hourly_kwh = hourly_sum_kw * 0.25

    hourly_df = pd.DataFrame({"kwh": hourly_kwh, "reading_count": hourly_count})
    hourly_df["incomplete"] = hourly_df["reading_count"] < 4

    # Apply DST flags
    dst_flags = dst_flag_series(hourly_df.index)
    hourly_df["dst"] = dst_flags["dst"]
    hourly_df["dst_march"] = dst_flags["dst_march"]
    hourly_df["dst_october"] = dst_flags["dst_october"]
    hourly_df["imputed"] = False
    hourly_df["scoreable"] = (~hourly_df["incomplete"]) & (~hourly_df["dst"])

    # Summary quality metrics
    quality = {
        "raw_readings": raw_len,
        "raw_start": str(start_raw),
        "raw_end": str(end_raw),
        "structural_zero_readings_trimmed": leading_zeros,
        "structural_zero_hours_trimmed": int(leading_zeros / 4),
        "active_hourly_records": len(hourly_df),
        "valid_scoreable_hours": int(hourly_df["scoreable"].sum()),
        "incomplete_hours_excluded": int(hourly_df["incomplete"].sum()),
        "dst_hours_flagged": int(hourly_df["dst"].sum()),
        "mean_kwh": float(hourly_df.loc[hourly_df["scoreable"], "kwh"].mean()),
        "std_kwh": float(hourly_df.loc[hourly_df["scoreable"], "kwh"].std()),
        "max_kwh": float(hourly_df.loc[hourly_df["scoreable"], "kwh"].max()),
        "min_kwh": float(hourly_df.loc[hourly_df["scoreable"], "kwh"].min()),
    }

    return hourly_df[["kwh", "incomplete", "dst", "dst_march", "dst_october", "imputed", "scoreable"]], quality


def preprocess_all(targets: list[str] | None = None):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    targets = targets or (SELECTED_10_CLIENTS + [PORTFOLIO])
    print(f"Beginning preprocessing for {len(targets)} targets from {REAL_TXT}...")

    # Read header to get column mapping
    header = pd.read_csv(REAL_TXT, sep=";", nrows=0).columns
    header_clean = [c.strip().strip('"').strip("'") for c in header]
    col_map = {name: idx for idx, name in enumerate(header_clean)}

    overall_quality = {}

    for target in targets:
        print(f"\nPreprocessing: {target}...")
        if target == PORTFOLIO:
            print("  Aggregating sum of all 370 client columns across 140,256 intervals...")
            chunks = []
            for ch in pd.read_csv(REAL_TXT, sep=";", decimal=",", chunksize=25000):
                ch.columns = [c.strip().strip('"').strip("'") for c in ch.columns]
                ts_col = ch.columns[0]
                client_cols = ch.columns[1:]
                ts = pd.to_datetime(ch[ts_col])
                kw_sum = ch[client_cols].astype(float).sum(axis=1)
                chunks.append(pd.Series(kw_sum.to_numpy(), index=ts))
            kw_s = pd.concat(chunks)
        else:
            col_idx = col_map[target]
            df_target = pd.read_csv(REAL_TXT, sep=";", decimal=",", usecols=[0, col_idx])
            df_target.columns = [c.strip().strip('"').strip("'") for c in df_target.columns]
            ts = pd.to_datetime(df_target.iloc[:, 0])
            kw_vals = df_target.iloc[:, 1].astype(float)
            kw_s = pd.Series(kw_vals.to_numpy(), index=ts)

        hourly_df, q = build_hourly_kwh(kw_s, trim_leading_zeros=(target != PORTFOLIO))
        out_csv = PROCESSED_DIR / f"{target}_hourly.csv"
        hourly_df.to_csv(out_csv)
        print(f"  Saved: {out_csv} ({len(hourly_df):,} hourly rows, {q['valid_scoreable_hours']:,} scoreable)")
        overall_quality[target] = q

    # Save data quality reports
    summary_report = {
        "preprocessing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "UCI ElectricityLoadDiagrams20112014",
        "power_to_energy_formula": "E_h = sum(P_15min) * 0.25 kWh",
        "targets_processed": len(targets),
        "target_summaries": overall_quality,
    }

    report_json_path = QUALITY_DIR / "data_quality_report.json"
    with open(report_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_report, fh, indent=2)

    rows = []
    for tgt, q in overall_quality.items():
        rows.append({
            "Target": tgt,
            "Valid Hourly Records": f"{q['active_hourly_records']:,}",
            "Scoreable Hours": f"{q['valid_scoreable_hours']:,}",
            "Trimmed Zero Hours": f"{q['structural_zero_hours_trimmed']:,}",
            "DST Hours Flagged": q["dst_hours_flagged"],
            "Mean kWh": f"{q['mean_kwh']:.2f}",
            "Std kWh": f"{q['std_kwh']:.2f}",
            "Max kWh": f"{q['max_kwh']:.2f}",
        })
    df_q = pd.DataFrame(rows)
    df_q.to_csv(QUALITY_DIR / "data_quality_summary.csv", index=False)

    md_path = QUALITY_DIR / "data_quality_report.md"
    md_text = f"""# GridSense: Data Quality & Preprocessing Assurance Report

**Execution Timestamp:** {summary_report['preprocessing_timestamp_utc'][:19]} UTC  
**Dataset Source:** UCI *ElectricityLoadDiagrams20112014* (Artur Trindade, DOI: 10.24432/C58C86)  
**Conversion Formulation:** $E_h = \\sum_{{j=1}}^4 P_{{h,j}} \\times 0.25\\text{{ kWh}}$  

---

## 1. Quality Summary Table

{df_q.to_markdown(index=False)}

---

## 2. Preprocessing & Quality Safeguards
1. **Physical Energy Conservation:** 15-minute average power (kW) converted to hourly energy (kWh) via multiplication by 0.25h.
2. **Incomplete Interval Defense:** Any hourly window with fewer than 4 readings is flagged `incomplete=True` and excluded from model loss calculation and test scoring.
3. **European Daylight Saving Time (DST):** March 23-hour transitions and October 25-hour transitions are flagged and isolated from model benchmarks.
4. **Leading Structural Zero Handling:** Pre-activation structural zeros are trimmed on a per-client basis without altering the underlying raw dataset.
"""
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)

    print(f"\nAll targets preprocessed! Quality report saved to: {md_path}")


if __name__ == "__main__":
    preprocess_all()
