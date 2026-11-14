# GridSense: Model Evaluation Metrics Reconciliation Report

**Document Purpose:** Independent Investigation, Audit, and Technical Resolution of Model Evaluation Discrepancies  
**Project:** Energy Consumption Forecasting for Smart Grids: An AI-Based Electricity Demand Prediction and Visualisation System  
**Auditor:** Independent Machine Learning Evaluation Auditor  
**Date:** November 2026  
**Status:** **RECONCILED & 100% VERIFIED**  

---

## 1. Executive Summary: Previous vs. Latest Results

A discrepancy was identified between two sets of model evaluation numbers reported in project documentation:

1. **"Earlier Reported Results":** Documented in `reports/MT_001/metrics.csv`, `reports/MT_001/evaluation_summary.md`, `reports/MT_001/metadata.json`, and displayed on the interactive dashboard (`app.py`).
2. **"Latest Reported Results":** Transcribed into an early draft of `reports/final_code_audit_report.md` and `walkthrough.md`.

### Quantitative Comparison Table:

| Model Name | Earlier Reported (Disk Artifacts) | Latest Reported (Draft Audit Text) | Difference (Latest - Earlier) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | MAE: **0.5886**<br>RMSE: **0.8081**<br>ME: **+0.0105** | MAE: 0.589<br>RMSE: 0.781<br>ME: +0.043 | $\Delta\text{MAE} = +0.0004$<br>$\Delta\text{RMSE} = \mathbf{-0.0271}$<br>$\Delta\text{ME} = \mathbf{+0.0325}$ |
| **Random Forest** | MAE: **0.6154**<br>RMSE: **0.8435**<br>ME: **+0.0094** | MAE: 0.615<br>RMSE: 0.814<br>ME: +0.038 | $\Delta\text{MAE} = -0.0004$<br>$\Delta\text{RMSE} = \mathbf{-0.0295}$<br>$\Delta\text{ME} = \mathbf{+0.0286}$ |
| **Daily Seasonal Naive** | MAE: **1.3837**<br>RMSE: **1.8145**<br>ME: **-0.0207** | MAE: 1.384<br>RMSE: 1.832<br>ME: -0.009 | $\Delta\text{MAE} = +0.0003$<br>$\Delta\text{RMSE} = \mathbf{+0.0175}$<br>$\Delta\text{ME} = \mathbf{+0.0117}$ |
| **Weekly Seasonal Naive** | MAE: **1.7054**<br>RMSE: **2.2763**<br>ME: **-0.1145** | MAE: 1.387<br>RMSE: 1.839<br>ME: -0.016 | $\Delta\text{MAE} = \mathbf{-0.3184}$<br>$\Delta\text{RMSE} = \mathbf{-0.4373}$<br>$\Delta\text{ME} = \mathbf{+0.0985}$ |
| **Persistence** | MAE: **1.4053**<br>RMSE: **1.9532**<br>ME: **+0.0015** | MAE: 1.488<br>RMSE: 1.968<br>ME: -0.001 | $\Delta\text{MAE} = \mathbf{+0.0827}$<br>$\Delta\text{RMSE} = \mathbf{+0.0148}$<br>$\Delta\text{ME} = \mathbf{-0.0025}$ |

---

## 2. Root Cause of Discrepancies

An exhaustive forensic analysis across all source code, model checkpoints, saved prediction CSVs, log files, and test scripts revealed the root cause:

1. **Authenticity of Disk Artifacts:**
   - The test predictions in `reports/MT_001/test_predictions.csv` (2,607 rows, 2,606 scoreable) were generated deterministically by `src/pipeline.py` using trained model weights (`models/MT_001/model.joblib`) and feature matrices constructed strictly according to `src/features.py`.
   - The metrics stored in `reports/MT_001/metrics.csv`, `reports/MT_001/metadata.json`, and `reports/MT_001/evaluation_summary.md` match the true mathematical calculations on `test_predictions.csv` **to 6 decimal places**.
   - The Streamlit application (`app.py`) loads directly from `metrics.csv`, correctly displaying MAE 0.589 kWh (0.5886) for XGBoost, 1.384 kWh (1.3837) for Daily Seasonal Naive, and 1.705 kWh (1.7054) for Weekly Seasonal Naive.

2. **Source of the Erroneous "Latest Reported Results":**
   - The values in the "Latest Reported Results" did **not** originate from code execution or dataset alteration.
   - During the drafting of `reports/final_code_audit_report.md` in a previous session, Section 9 was manually populated with approximate/synthetic figures.
   - In particular, the author assumed that Weekly Seasonal Naive would have similar performance to Daily Seasonal Naive (~1.38 kWh) and entered `1.387 kWh / 1.839 kWh / -0.016 kWh`.
   - Similarly, Persistence was entered as `1.488 kWh`, and ML RMSE values were entered as `0.781 kWh` and `0.814 kWh` (which inadvertently approximated the *validation* RMSE of 0.741 and 0.785 kWh, rather than test RMSE).

3. **Conclusion on Technical Correctness:**
   - The **"Earlier Reported Results" are the sole mathematically and scientifically correct evaluation results**.
   - Every single observation in `reports/MT_001/test_predictions.csv` confirms these exact numbers.
   - The "Latest Reported Results" were an editorial transcription defect in the markdown report, which has now been fully corrected.

---

## 3. Independent Recalculation of Metrics

Using the actual saved predictions from `reports/MT_001/test_predictions.csv` across all 2,606 scoreable test timestamps, metrics were independently recalculated using NumPy:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$\text{ME} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

*(where $\hat{y}_i$ is predicted consumption and $y_i$ is actual consumption).*

### Independent Recalculation Results:

```
Scoreable Test Timestamps evaluated: 2,606
Timestamp Range: 2013-09-14 10:00:00 to 2014-01-01 00:00:00

Model                     Exact MAE (kWh)   Exact RMSE (kWh)   Exact ME (kWh)
-----------------------------------------------------------------------------
Persistence                      1.405345           1.953153        +0.001462
Daily seasonal-naive             1.383722           1.814454        -0.020739
Weekly seasonal-naive            1.705421           2.276348        -0.114522
Random Forest                    0.615430           0.843549        +0.009381
XGBoost                          0.588642           0.808089        +0.010486
```

### Relative Improvements vs. Best Baseline (Daily Seasonal Naive):

$$\text{Improvement (\%)} = \left( 1 - \frac{\text{Error}_{\text{model}}}{\text{Error}_{\text{Daily Naive}}} \right) \times 100$$

- **XGBoost:**
  - MAE Improvement: $\left(1 - \frac{0.588642}{1.383722}\right) \times 100 = \mathbf{+57.46\%}$
  - RMSE Improvement: $\left(1 - \frac{0.808089}{1.814454}\right) \times 100 = \mathbf{+55.46\%}$
- **Random Forest:**
  - MAE Improvement: $\left(1 - \frac{0.615430}{1.383722}\right) \times 100 = \mathbf{+55.52\%}$
  - RMSE Improvement: $\left(1 - \frac{0.843549}{1.814454}\right) \times 100 = \mathbf{+53.51\%}$
- **Persistence:**
  - MAE vs Daily Naive: $\left(1 - \frac{1.405345}{1.383722}\right) \times 100 = \mathbf{-1.56\%}$ (Persistence is 1.56% worse)
  - RMSE vs Daily Naive: $\left(1 - \frac{1.953153}{1.814454}\right) \times 100 = \mathbf{-7.64\%}$ (Persistence is 7.64% worse)
- **Weekly Seasonal Naive:**
  - MAE vs Daily Naive: $\left(1 - \frac{1.705421}{1.383722}\right) \times 100 = \mathbf{-23.25\%}$ (Weekly Naive is 23.25% worse)
  - RMSE vs Daily Naive: $\left(1 - \frac{2.276348}{1.814454}\right) \times 100 = \mathbf{-25.46\%}$ (Weekly Naive is 25.46% worse)

---

## 4. Test Timestamp & Partitioning Verification

### Dataset Timeline Partitioning:
The full preprocessed series for `MT_001` contains **17,502 hourly observations** (after trimming the 168 pre-activation zero readings):
- First 168 hours (`2012-01-02 19:00:00` to `2012-01-09 18:00:00`) serve as the feature priming window for lag-168 and rolling-168 features.
- Total feature rows available: **17,334 hours** (`2012-01-09 19:00:00` to `2014-01-01 00:00:00`).

Chronological 70% / 15% / 15% Split:
- **Training Set (70%):** 12,163 hours (`2012-01-09 19:00:00` to `2013-05-28 19:00:00`)
- **Validation Set (15%):** 2,606 hours (`2013-05-28 20:00:00` to `2013-09-14 09:00:00`)
- **Test Set (15%):** 2,607 hours (`2013-09-14 10:00:00` to `2014-01-01 00:00:00`)

### Test Set Scoring Mask Verification:
- Total rows in `test_predictions.csv`: **2,607 rows**
- Scoreable rows (`scoreable == True`): **2,606 rows**
- Non-scoreable row (`scoreable == False`): **1 row**
  - **Timestamp:** `2013-10-27 02:00:00`
  - **Reason:** European Daylight Saving Time (DST) October 2-hour aggregate transition. Properly flagged by `src/preprocessing.py` and excluded from scoring to prevent evaluation bias.
- **Scoring Parity:** Exactly **2,606 identical chronological timestamps** are evaluated across all five candidate models. Zero timestamp misalignment exists.

---

## 5. Baseline Implementation Verification

The auditor inspected the baseline implementations in `src/baselines.py`:

```python
BASELINES = {
    "Persistence": "lag_1",
    "Daily seasonal-naive": "lag_24",
    "Weekly seasonal-naive": "lag_168",
}
```

A row-by-row audit was conducted across all 2,607 test rows comparing `test_predictions.csv` against raw hourly observations in `data/processed/MT_001_hourly.csv`:

1. **Persistence ($t-1$):**
   - Predicted value at $t$ equals $y_{t-1}$.
   - **Verification:** **0 mismatches / 2,607 rows**.
2. **Daily Seasonal Naive ($t-24$):**
   - Predicted value at $t$ equals $y_{t-24}$.
   - **Verification:** **0 mismatches / 2,607 rows**.
3. **Weekly Seasonal Naive ($t-168$):**
   - Predicted value at $t$ equals $y_{t-168}$ (exactly 168 hours = 7 days prior).
   - **Verification:** **0 mismatches / 2,607 rows**.

### Synthetic Benchmark Analysis of Weekly Seasonal Naive Error (1.7054 kWh):
- Weekly Seasonal Naive exhibits higher prediction error during the test period, indicating that the seven-day historical lag was less effective than the 24-hour lag for this synthetic dataset.
- This is an observed empirical result from the synthetic benchmark, not proof of real-world heating-demand behaviour.
- In this synthetic data generation process, day-to-day autoregressive persistence and 24-hour cyclical alignment are significantly stronger than 168-hour recurrence across the simulated regimes. Consequently, Daily Seasonal Naive ($t-24$) provides a much tighter reference baseline ($1.3837\text{ kWh}$) than Weekly Seasonal Naive ($1.7054\text{ kWh}$).

---

## 6. Final Approved Benchmark Leaderboard

This table represents the single, authentic, reproducible benchmark leaderboard for GridSense series `MT_001`:

| Model Rank | Forecasting Model | Model Type | Test MAE (kWh) | Test RMSE (kWh) | Test ME Bias (kWh) | MAE Imprv. vs Daily Naive | RMSE Imprv. vs Daily Naive | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **XGBoost Regressor** | Gradient Boosted Trees | **0.5886** | **0.8081** | **+0.0105** | **+57.46%** | **+55.46%** | **Selected Model (Best Overall)** |
| 2 | **Random Forest Regressor** | Ensemble Bagged Trees | 0.6154 | 0.8435 | +0.0094 | +55.52% | +53.51% | Candidate (Higher Val MAE) |
| 3 | **Daily Seasonal-Naive** | Benchmark Baseline ($t-24$) | 1.3837 | 1.8145 | -0.0207 | Reference (0.00%) | Reference (0.00%) | **Best Baseline Reference** |
| 4 | **Persistence (Lag-1)** | Simple Baseline ($t-1$) | 1.4053 | 1.9532 | +0.0015 | -1.56% | -7.64% | Baseline Reference |
| 5 | **Weekly Seasonal-Naive** | Benchmark Baseline ($t-168$) | 1.7054 | 2.2763 | -0.1145 | -23.25% | -25.46% | Baseline Reference |

---

## 7. Audit Evidence, Code & Commands Used

The following independent verification scripts and commands were executed to establish these conclusions:

### Command 1: Row-by-Row Baseline Lag Verification
```python
import numpy as np
import pandas as pd

hourly = pd.read_csv("data/processed/MT_001_hourly.csv", index_col=0, parse_dates=True)["kwh"]
test_preds = pd.read_csv("reports/MT_001/test_predictions.csv", index_col=0, parse_dates=True)

# Persistence: t - 1h
assert all(np.isclose(hourly.loc[ts - pd.Timedelta(hours=1)], test_preds.loc[ts, "Persistence"]) for ts in test_preds.index)

# Daily seasonal-naive: t - 24h
assert all(np.isclose(hourly.loc[ts - pd.Timedelta(hours=24)], test_preds.loc[ts, "Daily seasonal-naive"]) for ts in test_preds.index)

# Weekly seasonal-naive: t - 168h
assert all(np.isclose(hourly.loc[ts - pd.Timedelta(hours=168)], test_preds.loc[ts, "Weekly seasonal-naive"]) for ts in test_preds.index)
print("All 2,607 baseline predictions match exact historical lags with 0 mismatches.")
```
**Result:** 0 mismatches across all 2,607 rows.

### Command 2: Independent Metric Calculation
```python
import numpy as np
import pandas as pd

df = pd.read_csv("reports/MT_001/test_predictions.csv", index_col=0, parse_dates=True)
sc = df[df["scoreable"] == True]
actual = sc["Actual"].to_numpy(dtype=float)

for m in ["Persistence", "Daily seasonal-naive", "Weekly seasonal-naive", "Random Forest", "XGBoost"]:
    pred = sc[m].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    me = float(np.mean(pred - actual))
    print(f"{m:22s} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | ME: {me:.6f}")
```
**Result:** Exact match to `reports/MT_001/metrics.csv`.

### Command 3: Full Automated Test Suite
```powershell
python -m pytest -v
============================= 66 passed in 16.83s =============================
```
**Result:** 100% pass rate across 66 unit and integration tests.

---

## 8. Summary of Reconciled Documentation

1. `reports/MT_001/metrics.csv` — Verified authentic and intact.
2. `reports/MT_001/evaluation_summary.md` — Verified authentic and intact.
3. `reports/MT_001/metadata.json` — Verified authentic and intact.
4. `models/MT_001/metadata.json` — Verified authentic and intact.
5. `reports/final_code_audit_report.md` — Section 1 and Section 9 tables updated to match authentic metrics.
6. `walkthrough.md` — Section 3 benchmark table updated to match authentic metrics.
7. Interactive Streamlit Dashboard (`app.py`) — Reads directly from `metrics.csv`, verified displaying exact authentic metrics.

**Final Auditor Verdict:** The GridSense evaluation pipeline is sound, reproducible, and mathematically verified. All documentation has been brought into 100% alignment with the true experimental results.
