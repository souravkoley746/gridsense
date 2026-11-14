"""Generate a SYNTHETIC demo dataset in the exact UCI ElectricityLoadDiagrams20112014 layout.

  * semicolon separated, decimal comma, quoted empty first header
  * 15-minute kW readings; timestamps are interval-end labels
  * some clients "created later" -> leading structural zeros
  * DST quirks reproduced as documented by UCI: March change-day hour = zeros,
    October change-day hour = aggregate of two hours

The numbers are simulated (temperature-driven load + daily/weekly cycles + noise). They are NOT
real consumption data. For genuine experiments download LD2011_2014.txt from the UCI repository
and point config/experiment.json at it.

Usage: python scripts/generate_demo_dataset.py [--out data/raw/LD2011_2014_demo.txt]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# name: (type, scale kW, start date or None)
CLIENTS = {
    "MT_001": ("residential", 14, None),
    "MT_002": ("residential", 22, None),
    "MT_003": ("commercial", 95, None),
    "MT_004": ("commercial", 160, None),
    "MT_005": ("industrial", 420, None),
    "MT_006": ("industrial", 760, None),
    "MT_007": ("commercial", 70, "2012-04-10"),      # created later -> structural zeros
    "MT_008": ("residential", 18, "2012-09-01"),     # created later -> structural zeros
}
HOLIDAYS = ["01-01", "04-25", "05-01", "06-10", "08-15", "10-05", "11-01", "12-01", "12-08", "12-25"]


def _profile(kind: str, hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    h = hour.astype(float)
    if kind == "residential":
        wd = 0.55 + 0.35 * np.exp(-((h - 7.5) ** 2) / 4.5) + 0.75 * np.exp(-((h - 20.5) ** 2) / 6.0) + 0.15 * np.exp(-((h - 13) ** 2) / 8)
        we = 0.6 + 0.30 * np.exp(-((h - 10) ** 2) / 12) + 0.7 * np.exp(-((h - 20.5) ** 2) / 8.0)
    elif kind == "commercial":
        plateau = 1 / (1 + np.exp(-(h - 7.5) * 1.6)) - 1 / (1 + np.exp(-(h - 19) * 1.4))
        wd = 0.28 + 0.9 * plateau
        we = 0.25 + 0.30 * plateau
    else:  # industrial: near-continuous with shift pattern
        plateau = 1 / (1 + np.exp(-(h - 5.5) * 1.2)) - 1 / (1 + np.exp(-(h - 22) * 1.2))
        wd = 0.55 + 0.45 * plateau
        we = 0.45 + 0.15 * plateau
    return np.where(weekend, we, wd)


def simulate(seed: int = 2024, start="2012-01-01", end="2013-12-31 23:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h")
    doy = hours.dayofyear.to_numpy()
    # temperature proxy: seasonal + diurnal + AR(1) weather noise (Southern Europe like)
    seasonal = 16 + 8.5 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    diurnal = 4.0 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 9) / 24)
    ar = np.zeros(len(hours))
    for i in range(1, len(hours)):
        ar[i] = 0.985 * ar[i - 1] + rng.normal(0, 0.55)
    temp = seasonal + diurnal + ar
    heat = np.clip(15 - temp, 0, None) / 10
    cool = np.clip(temp - 22, 0, None) / 10
    weekend = hours.dayofweek.to_numpy() >= 5
    mmdd = hours.strftime("%m-%d")
    holiday = np.isin(mmdd, HOLIDAYS)

    out = {}
    for name, (kind, scale, created) in CLIENTS.items():
        base = _profile(kind, hours.hour.to_numpy(), weekend | holiday)
        sens = {"residential": (0.55, 0.45), "commercial": (0.25, 0.60), "industrial": (0.06, 0.10)}[kind]
        level = base * (1 + sens[0] * heat + sens[1] * cool)
        slow = np.zeros(len(hours))
        for i in range(1, len(hours)):
            slow[i] = 0.9 * slow[i - 1] + rng.normal(0, 0.012 if kind == "industrial" else 0.03)
        noise = rng.normal(0, 0.025, len(hours))
        series = scale * level * (1 + slow + noise) * (1 + 0.06 * np.sin(2 * np.pi * np.arange(len(hours)) / (24 * 365.25)))
        if kind == "industrial":  # occasional maintenance days
            for d in rng.choice(np.unique(hours.normalize()), 5, replace=False):
                series[hours.normalize() == d] *= 0.45
        # rare demand spikes
        spikes = rng.random(len(hours)) < 0.0015
        series[spikes] *= rng.uniform(1.25, 1.6, spikes.sum())
        out[name] = np.clip(series, 0, None)
    hourly_kw = pd.DataFrame(out, index=hours)

    # to 15-minute resolution with interval-END labels; first label = first hour start + 15 min
    idx15 = pd.date_range(hours[0] + pd.Timedelta(minutes=15), hours[-1] + pd.Timedelta(hours=1), freq="15min")
    base_idx = hourly_kw.index
    arr = {}
    for c in hourly_kw.columns:
        v = hourly_kw[c].reindex(base_idx.append(pd.DatetimeIndex([base_idx[-1] + pd.Timedelta(hours=1)]))).ffill()
        v15 = np.repeat(v.to_numpy()[:-1], 4)
        v15 = v15 * (1 + rng.normal(0, 0.015, len(v15)))
        arr[c] = v15
    df = pd.DataFrame(arr, index=idx15[: len(next(iter(arr.values())))])

    # clients created later -> leading zeros (structural zeros)
    for name, (_, _, created) in CLIENTS.items():
        if created:
            df.loc[df.index < pd.Timestamp(created), name] = 0.0

    # DST quirks (UCI): last Sunday of March -> hour 01:00-02:00 zeros; October -> aggregate of two hours
    def last_sunday(y, m):
        d = pd.Timestamp(year=y, month=m, day=31)
        return (d - pd.Timedelta(days=(d.dayofweek + 1) % 7)).normalize()
    for y in (2012, 2013):
        a = last_sunday(y, 3) + pd.Timedelta(hours=1)
        mask = (df.index > a) & (df.index <= a + pd.Timedelta(hours=1))
        df.loc[mask, :] = 0.0
        b = last_sunday(y, 10) + pd.Timedelta(hours=1)
        mask = (df.index > b) & (df.index <= b + pd.Timedelta(hours=1))
        df.loc[mask, :] = df.loc[mask, :] * 2.0
    return df.round(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "LD2011_2014_demo.txt"))
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    df = simulate(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = ""
    df.to_csv(out, sep=";", decimal=",", float_format="%.6f", quoting=1)
    print(f"wrote {out}  rows={len(df):,} clients={df.shape[1]}  size={out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
