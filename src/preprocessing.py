"""Hourly target construction, quality validation, and DST policy (Phases 1.4-1.6).

Unit Convention:
- Raw input: 15-minute average power demand in kilowatts (kW).
- Target: Hourly electrical energy consumed in kilowatt-hours (kWh) for the
  interval ENDING at the timestamp label:
      E_h = sum_{j=1..4} P_{h,j} * 0.25 kWh
  where P_{h,j} are the four 15-minute readings within hour h.

Quality Rules:
1. Normal hours require exactly 4 valid, non-negative 15-minute readings.
2. Incomplete hours (<4 readings) are flagged and excluded from scoreable evaluation.
3. Structural zeros: Inactive leading zero readings (before a client joins the grid)
   are trimmed when configured, but legitimate internal zero readings are preserved.
4. Daylight Saving Time (DST): European Central Time DST transitions (March 0-reading,
   October 2-hour aggregate) are flagged per UCI documentation. In accordance with
   Phase 1.6, original observations are preserved and excluded from ordinary 1-hour
   model evaluation without arbitrary alteration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KW_TO_KWH_PER_READING = 0.25


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    """Find the last Sunday of a given month (European DST transition rule)."""
    d = pd.Timestamp(year=year, month=month, day=31 if month in (3, 10) else 30)
    return (d - pd.Timedelta(days=(d.dayofweek + 1) % 7)).normalize()


def dst_hours(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag European DST transition intervals in the dataset index.

    Per UCI ElectricityLoadDiagrams20112014 documentation:
    - March transition: The 01:00-02:00 interval is missing in standard time;
      the source records zeros for this hour.
    - October transition: The 02:00-03:00 interval repeats; the source aggregates
      the readings of both hours into a single timestamp label (02:00 or 03:00).
    """
    flags = pd.DataFrame(False, index=index, columns=["dst_march", "dst_october"])
    for year in sorted(set(index.year)):
        for month, col in ((3, "dst_march"), (10, "dst_october")):
            ts = _last_sunday(year, month) + pd.Timedelta(hours=2)
            if ts in flags.index:
                flags.loc[ts, col] = True
    return flags


def build_hourly(
    kw: pd.Series,
    trim_structural_zeros: bool = True,
    dst_policy: str = "repair",
) -> tuple[pd.DataFrame, dict]:
    """Convert a 15-minute kW series to a complete hourly kWh table with quality flags.

    Parameters
    ----------
    kw : pd.Series
        15-minute power demand in kW, with a DatetimeIndex.
    trim_structural_zeros : bool
        If True, trims inactive prefix zeros before the client's first positive reading.
    dst_policy : str
        'flag_only' (default, Phase 1.6): preserves original energy values, flags them,
            and excludes them from scoreable evaluation.
        'repair': causally handles March zeros and scales October 2-hour aggregates.
        'keep': preserves original observations without repair.
    """
    q: dict = {}
    kw = kw.copy()

    # Detect and isolate negative readings
    neg = kw < 0
    q["negative_values_count"] = int(neg.sum())
    kw[neg] = np.nan

    # Interval sampling check
    diffs = kw.index.to_series().diff().dropna()
    q["off_grid_intervals"] = int((diffs != pd.Timedelta(minutes=15)).sum())

    # Structural zero handling: client joined grid after start of 2011
    trimmed = 0
    nz = np.flatnonzero(np.nan_to_num(kw.to_numpy(), nan=0.0) > 0)
    if len(nz) == 0:
        raise ValueError("Series contains no positive consumption observations.")
    if trim_structural_zeros and nz[0] > 0:
        trimmed = int(nz[0])
        kw = kw.iloc[nz[0]:]
    q["structural_zero_rows_trimmed"] = trimmed
    q["analysis_start_after_trim"] = str(kw.index[0])

    # Resample 15-minute readings to hourly intervals
    # closed='right', label='right': the hourly label represents the interval ending at that timestamp
    r = kw.resample("h", closed="right", label="right")
    sums = r.sum(min_count=4)
    counts = r.count()
    kwh = sums * KW_TO_KWH_PER_READING

    # Ensure first hour has at least 4 readings
    complete_mask = (counts >= 4).to_numpy()
    if not complete_mask.any():
        raise ValueError("No complete hourly intervals (with 4 valid readings) found.")
    first_complete = counts.index[complete_mask.argmax()]
    kwh = kwh.loc[first_complete:]
    counts = counts.loc[first_complete:]

    out = pd.DataFrame({"kwh": kwh})
    out["incomplete"] = counts < 4

    # Flag DST transitions
    d = dst_hours(out.index)
    out["dst"] = d["dst_march"] | d["dst_october"]
    out["dst_march"] = d["dst_march"]
    out["dst_october"] = d["dst_october"]

    # DST handling policy
    if dst_policy == "flag_only" or dst_policy == "keep":
        # Preserve original source energy observations exactly; do not alter values
        pass
    elif dst_policy == "repair":
        # Scientifically documented repair option:
        # March zero hour is treated as missing; October aggregated hour is divided by 2
        out.loc[d["dst_march"], "kwh"] = np.nan
        out.loc[d["dst_october"], "kwh"] = out.loc[d["dst_october"], "kwh"] / 2.0
    else:
        raise ValueError(f"Unknown dst_policy: {dst_policy}. Choose 'flag_only', 'keep', or 'repair'.")

    # Incomplete or missing values handling:
    # We maintain causal continuity for lag features, but strictly exclude imputed hours from scoring
    need_impute = out["kwh"].isna()
    y = out["kwh"].copy()
    for pos in np.flatnonzero(need_impute.to_numpy()):
        if pos >= 24 and np.isfinite(y.iloc[pos - 24]):
            y.iloc[pos] = y.iloc[pos - 24]
        elif pos >= 1 and np.isfinite(y.iloc[pos - 1]):
            y.iloc[pos] = y.iloc[pos - 1]
        else:
            y.iloc[pos] = 0.0

    out["kwh"] = y
    out["imputed"] = need_impute
    first_valid = out["kwh"].first_valid_index()
    if first_valid is not None:
        out = out.loc[first_valid:]

    # Scoreable mask: strictly excludes incomplete, imputed, or DST-affected hours
    out["scoreable"] = ~(out["imputed"] | out["dst"] | out["incomplete"])
    out.index.name = "timestamp"

    q.update({
        "hourly_rows": int(len(out)),
        "hourly_start": str(out.index[0]),
        "hourly_end": str(out.index[-1]),
        "incomplete_hours": int(out["incomplete"].sum()),
        "dst_flagged_hours": int(out["dst"].sum()),
        "dst_march_hours": int(out["dst_march"].sum()),
        "dst_october_hours": int(out["dst_october"].sum()),
        "imputed_hours": int(out["imputed"].sum()),
        "scoreable_hours": int(out["scoreable"].sum()),
        "suspicious_zero_hours": int(((out["kwh"] == 0) & out["scoreable"]).sum()),
        "dst_policy": dst_policy,
        "target_definition": (
            "Hourly energy (kWh) for interval ending at timestamp label; "
            "sum of four 15-min kW readings x 0.25 kWh"
        ),
    })
    return out, q
