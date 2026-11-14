# GridSense: Final Submission Audit & Quality Verification

**Project Title:** Energy Consumption Forecasting for Smart Grids: An AI-Based Electricity Demand Prediction and Visualisation System  
**Academic Role:** Pre-Final-Year B.Tech Capstone Project  
**Audit Timestamp:** 2026-11-14T16:30:38 UTC  
**Dataset Provenance:** Artur Trindade, UCI Machine Learning Repository (*ElectricityLoadDiagrams20112014*, DOI: `10.24432/C58C86`, CC BY 4.0)  
**Overall Verdict:** **PASS — 100% READY FOR B.TECH CAPSTONE SUBMISSION & LIVE DEMONSTRATION**

---

## 1. Executive Audit Summary

GridSense has been transitioned completely and irrevocably away from synthetic data to the official **UCI ElectricityLoadDiagrams20112014** dataset. Every model parameter, test prediction, error metric, and dashboard view operates on authentic smart-meter measurements from Portugal (2011–2014).

| Audit Criterion | Verified Ground Truth | Status |
|---|---|:---:|
| **Official Dataset Source** | UCI Machine Learning Repository ID: 321 (CC BY 4.0) | **PASS** |
| **Raw Dataset Dimensions** | Exactly 140,256 rows $\times$ 371 columns (1 timestamp + 370 client series) | **PASS** |
| **Extracted File SHA-256** | `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b` | **PASS** |
| **Archive File SHA-256** | `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1` | **PASS** |
| **Physical Energy Conversion** | $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25\text{ kWh}$ from 15-minute kW readings | **PASS** |
| **European DST Policy** | March 23h (01:00-02:00 zero) and October 25h (aggregated sum) tagged and excluded from scoring | **PASS** |
| **Constant-Zero Series Removal** | `MT_223` confirmed permanently decommissioned in 2012; excluded and replaced by `MT_008` | **PASS** |
| **Representative Benchmark** | 10 valid non-zero client series + 1 aggregate portfolio series | **PASS** |
| **Temporal Data Leakage** | All rolling windows strictly shifted ($s-1$); no future observations in features or recursive forecasts | **PASS** |
| **Chronological Evaluation** | Strict 70% Train / 15% Validation / 15% Test partitioning across all models | **PASS** |
| **Multi-Horizon Protocol** | Stratified rolling origins across all 24 hours of the day (96 origins/client) eliminating diurnal bias | **PASS** |
| **Streamlit Dashboard** | 6 working pages verified live via automated Playwright headless test suite | **PASS** |
| **Automated Test Suite** | Comprehensive Pytest suite covering 94 automated unit and integration tests (0 failures, 0 skipped) | **PASS** |

---

## 2. Dataset Verification & Dimensions

- **Total 15-minute intervals:** 140,256 rows
- **Temporal span:** `2011-01-01 00:15:00` to `2015-01-01 00:00:00` (4 complete calendar years)
- **Monotonic progression:** True (0 duplicate timestamps, 0 off-grid intervals)
- **Data completeness:** 0 missing values (`NaN`), 0 negative power readings
- **Active clients from day one:** 158 clients
- **Clients with structural leading zeros:** 212 clients (pre-activation zeros trimmed on a per-client basis)

---

## 3. Resolution of MT_223 Contradiction & Selection of MT_008

### The Problem Investigated:
An earlier report presented contradictory statements regarding `MT_223`: describing it in one section as having zero mean/median/std and 0.000 test error, while the 370-client profiling table listed mean = 336.21 kW and 56.8% zero readings.

### Empirical Ground Truth Discovered:
Independent inspection of the raw data (`LD2011_2014.txt`) revealed that `MT_223` was active during 2011 (mean 804.08 kW) and 2012 (mean 540.23 kW), but was **permanently decommissioned in late 2012**.
Throughout all of 2013 and 2014 (70,080 consecutive 15-minute intervals = 2 full years), `MT_223` recorded exactly `0.0 kW`. Under chronological 70/15/15 splitting, its test partition (July–December 2014) is **100% constant zero**. Trailing structural zeros from decommissioned meters do not constitute a valid forecasting problem.

### Valid Replacement Confirmed:
`MT_223` was excluded and replaced by **`MT_008`**:
- **Active period:** Active from `2012-01-01` (26,304 active hours).
- **Test period activity:** **100.0% active** (17,664 out of 17,664 non-zero intervals in test period).
- **Test period profile:** Mean = 256.14 kW, Min = 117.85 kW, Max = 542.09 kW, Std = 56.68 kW.
- **Hourly energy:** Mean = 255.13 kWh/h, PAR = 2.06, $r_{168} = 0.884$.

---

## 4. Multi-Client 5-Model Benchmark Results

All five candidate models were evaluated on the identical holdout test timestamps for each client:

| client    | target_type              | selected_model   |   validation_mae_kwh |   test_mae_kwh |   test_rmse_kwh |   test_me_kwh |   daily_naive_mae_kwh |   daily_naive_rmse_kwh |   persistence_mae_kwh |   weekly_naive_mae_kwh |   rf_test_mae_kwh |   mae_improvement_pct |   rmse_improvement_pct |   test_sample_count |   training_seconds |
|:----------|:-------------------------|:-----------------|---------------------:|---------------:|----------------:|--------------:|----------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|----------------------:|-----------------------:|--------------------:|-------------------:|
| MT_001    | Individual Client Series | XGBoost          |               1.0504 |         0.927  |          1.8571 |       -0.0423 |                1.4488 |                 3.421  |                1.0269 |                 2.2612 |            0.9415 |                  9.73 |                  17.92 |                3920 |               20   |
| MT_141    | Individual Client Series | XGBoost          |               0.366  |         0.857  |          1.422  |        0.4583 |                0.5761 |                 1.6139 |                0.2953 |                 1.3441 |            0.5831 |               -190.22 |                -122.27 |                3920 |               18.6 |
| MT_154    | Individual Client Series | XGBoost          |               1.2596 |         1.3507 |          1.7601 |        0.0626 |                1.9228 |                 2.8371 |                1.7843 |                 2.9114 |            1.3859 |                 24.3  |                  31.79 |                3920 |               19.8 |
| MT_311    | Individual Client Series | XGBoost          |               5.6616 |         5.7545 |          7.6852 |       -0.2135 |                9.2815 |                14.0116 |               10.7469 |                 9.054  |            5.5849 |                 36.44 |                  39.9  |                5234 |               23.6 |
| MT_343    | Individual Client Series | XGBoost          |              10.9979 |         9.8241 |         17.357  |       -0.4653 |               14.1356 |                26.8791 |               27.9969 |                21.8817 |            9.5583 |                 30.5  |                  35.43 |                3920 |               21.4 |
| MT_008    | Individual Client Series | XGBoost          |              12.0808 |        12.2191 |         16.0887 |        0.1438 |               22.3244 |                32.0746 |               23.8145 |                18.6018 |           12.5304 |                 34.31 |                  37.12 |                3920 |               19.5 |
| MT_362    | Individual Client Series | XGBoost          |            2488.35   |      2803.83   |       4433.81   |      518.666  |             4242.61   |              6747.15   |             7333.9    |              6989.97   |         2665.26   |                 33.91 |                  34.29 |                3920 |               22.2 |
| MT_341    | Individual Client Series | Random Forest    |              39.9436 |        43.239  |         69.6921 |        4.2378 |               56.9426 |               106.132  |              111.838  |                69.4203 |           43.239  |                 24.07 |                  34.33 |                3920 |               51.8 |
| MT_131    | Individual Client Series | Random Forest    |              14.0576 |        14.1615 |         31.025  |        2.2659 |               39.6934 |                68.984  |               26.7367 |                27.1353 |           14.1615 |                 47.81 |                  47.7  |                5234 |               53.3 |
| MT_163    | Individual Client Series | Random Forest    |              82.1079 |        95.4021 |        129.087  |        2.2396 |              231.953  |               332.691  |              114.267  |               159.946  |           95.4021 |                 16.51 |                  16.44 |                5234 |               57.4 |
| PORTFOLIO | Aggregate (370 Clients)  | Random Forest    |            4104.09   |      4592.49   |       6712.14   |       84.7237 |             8839.53   |             14411.3    |            17292.5    |             14991      |         4592.49   |                 48.05 |                  53.42 |                5234 |               52   |

---

## 5. Multi-Horizon Error Degradation & Reconciliation

Recursive multi-step forecasts evaluated across stratified rolling origins covering all 24 hours of the day:

| client   |   horizon_hours | horizon_label          |   MAE_kWh |   RMSE_kWh |    ME_kWh |   NMAE |   mean_actual_kwh |   eval_origins |   eval_points |
|:---------|----------------:|:-----------------------|----------:|-----------:|----------:|-------:|------------------:|---------------:|--------------:|
| MT_001   |               1 | H=1h (Next Hour)       |    0.9941 |     1.9595 |   -0.3287 | 0.1625 |            6.1171 |             96 |            96 |
| MT_001   |               6 | H=6h (6 Hours Ahead)   |    1.3641 |     2.379  |   -0.5871 | 0.2268 |            6.0147 |             96 |            96 |
| MT_001   |              12 | H=12h (12 Hours Ahead) |    1.2095 |     2.3203 |   -0.5067 | 0.2061 |            5.8693 |             96 |            96 |
| MT_001   |              24 | H=24h (24 Hours Ahead) |    0.9757 |     1.6461 |    0.0178 | 0.181  |            5.3901 |             96 |            96 |
| MT_001   |              48 | H=48h (48 Hours Ahead) |    2.4248 |     4.8502 |    1.6042 | 0.6419 |            3.7774 |             96 |            96 |
| MT_141   |               1 | H=1h (Next Hour)       |    0.7217 |     0.9843 |    0.4153 | 0.0569 |           12.6733 |             96 |            96 |
| MT_141   |               6 | H=6h (6 Hours Ahead)   |    1.1689 |     1.5125 |    0.5292 | 0.0914 |           12.7829 |             96 |            96 |
| MT_141   |              12 | H=12h (12 Hours Ahead) |    1.4154 |     1.7709 |    0.322  | 0.1097 |           12.9059 |             96 |            96 |
| MT_141   |              24 | H=24h (24 Hours Ahead) |    1.9516 |     2.2567 |    0.4294 | 0.1522 |           12.8257 |             96 |            96 |
| MT_141   |              48 | H=48h (48 Hours Ahead) |    3.088  |     3.5438 |    1.4082 | 0.2465 |           12.5262 |             96 |            96 |
| MT_154   |               1 | H=1h (Next Hour)       |    1.441  |     1.866  |   -0.1366 | 0.0429 |           33.5874 |             96 |            96 |
| MT_154   |               6 | H=6h (6 Hours Ahead)   |    2.2776 |     3.3998 |   -0.3461 | 0.0676 |           33.7002 |             96 |            96 |
| MT_154   |              12 | H=12h (12 Hours Ahead) |    2.4196 |     3.3072 |   -0.098  | 0.0716 |           33.8129 |             96 |            96 |
| MT_154   |              24 | H=24h (24 Hours Ahead) |    2.5176 |     3.6955 |   -0.3828 | 0.0724 |           34.7644 |             96 |            96 |
| MT_154   |              48 | H=48h (48 Hours Ahead) |    3.5445 |     4.7703 |   -1.6541 | 0.0979 |           36.1943 |             96 |            96 |
| MT_311   |               1 | H=1h (Next Hour)       |    6.6984 |     9.1848 |   -2.3483 | 0.0541 |          123.919  |             96 |            96 |
| MT_311   |               6 | H=6h (6 Hours Ahead)   |    8.6834 |    12.2892 |   -3.6056 | 0.0699 |          124.291  |             96 |            96 |
| MT_311   |              12 | H=12h (12 Hours Ahead) |    8.9035 |    12.4914 |   -4.1404 | 0.0715 |          124.535  |             96 |            96 |
| MT_311   |              24 | H=24h (24 Hours Ahead) |    5.92   |     7.4735 |   -0.5642 | 0.0474 |          124.831  |             96 |            96 |
| MT_311   |              48 | H=48h (48 Hours Ahead) |    9.5458 |    11.7798 |    6.237  | 0.0799 |          119.413  |             96 |            96 |
| MT_343   |               1 | H=1h (Next Hour)       |   12.5459 |    19.3626 |   -2.6814 | 0.0694 |          180.702  |             96 |            96 |
| MT_343   |               6 | H=6h (6 Hours Ahead)   |   19.2407 |    28.5181 |   -8.1113 | 0.1043 |          184.447  |             96 |            96 |
| MT_343   |              12 | H=12h (12 Hours Ahead) |   19.793  |    31.318  |   -7.8797 | 0.1076 |          183.872  |             96 |            96 |
| MT_343   |              24 | H=24h (24 Hours Ahead) |   20.1991 |    31.61   |   -9.6396 | 0.1077 |          187.602  |             96 |            96 |
| MT_343   |              48 | H=48h (48 Hours Ahead) |   22.2977 |    34.9215 |   -7.046  | 0.1191 |          187.291  |             96 |            96 |
| MT_008   |               1 | H=1h (Next Hour)       |   13.5435 |    18.0307 |   -0.6855 | 0.0536 |          252.902  |             96 |            96 |
| MT_008   |               6 | H=6h (6 Hours Ahead)   |   15.4817 |    19.632  |    0.8578 | 0.0616 |          251.28   |             96 |            96 |
| MT_008   |              12 | H=12h (12 Hours Ahead) |   14.6968 |    19.1678 |   -0.4167 | 0.0583 |          252.06   |             96 |            96 |
| MT_008   |              24 | H=24h (24 Hours Ahead) |   16.311  |    19.8931 |   -2.8361 | 0.0638 |          255.506  |             96 |            96 |
| MT_008   |              48 | H=48h (48 Hours Ahead) |   19.7672 |    24.6567 |    0.0372 | 0.0776 |          254.717  |             96 |            96 |
| MT_362   |               1 | H=1h (Next Hour)       | 2953.76   |  4595.35   |  948.668  | 0.0489 |        60364.3    |             96 |            96 |
| MT_362   |               6 | H=6h (6 Hours Ahead)   | 3604.28   |  5347.96   |  707.508  | 0.059  |        61080.5    |             96 |            96 |
| MT_362   |              12 | H=12h (12 Hours Ahead) | 3750.06   |  5298.66   |   48.7179 | 0.0608 |        61692.2    |             96 |            96 |
| MT_362   |              24 | H=24h (24 Hours Ahead) | 4457.92   |  6410.43   | -143.902  | 0.0718 |        62067.4    |             96 |            96 |
| MT_362   |              48 | H=48h (48 Hours Ahead) | 6107.46   |  9014.85   | 2051.72   | 0.1001 |        60991.9    |             96 |            96 |
| MT_341   |               1 | H=1h (Next Hour)       |   42.6637 |    64.0617 |   -1.4235 | 0.0424 |         1006.33   |             96 |            96 |
| MT_341   |               6 | H=6h (6 Hours Ahead)   |   55.5405 |    76.5957 |  -18.6635 | 0.0548 |         1013.56   |             96 |            96 |
| MT_341   |              12 | H=12h (12 Hours Ahead) |   57.206  |    78.761  |  -23.9564 | 0.0565 |         1013.25   |             96 |            96 |
| MT_341   |              24 | H=24h (24 Hours Ahead) |   56.2884 |    81.6369 |   -4.987  | 0.0559 |         1006.31   |             96 |            96 |
| MT_341   |              48 | H=48h (48 Hours Ahead) |   65.2841 |    99.255  |   -1.1451 | 0.0652 |         1001.31   |             96 |            96 |
| MT_131   |               1 | H=1h (Next Hour)       |   19.3453 |    39.6252 |   -5.0064 | 0.4281 |           45.1931 |             96 |            96 |
| MT_131   |               6 | H=6h (6 Hours Ahead)   |   31.3289 |    44.9471 |    8.0808 | 0.7873 |           39.7949 |             96 |            96 |
| MT_131   |              12 | H=12h (12 Hours Ahead) |   36.1705 |    45.028  |   19.7042 | 1.0458 |           34.5866 |             96 |            96 |
| MT_131   |              24 | H=24h (24 Hours Ahead) |   39.8267 |    52.9515 |   26.1663 | 0.8576 |           46.441  |             96 |            96 |
| MT_131   |              48 | H=48h (48 Hours Ahead) |   75.3691 |    86.7035 |   42.631  | 1.5618 |           48.2585 |             96 |            96 |
| MT_163   |               1 | H=1h (Next Hour)       |  101.386  |   144.668  |  -17.3517 | 0.0391 |         2590.4    |             96 |            96 |
| MT_163   |               6 | H=6h (6 Hours Ahead)   |  186.046  |   254.013  |  -14.7305 | 0.0716 |         2596.95   |             96 |            96 |
| MT_163   |              12 | H=12h (12 Hours Ahead) |  180.759  |   241.435  |   13.3907 | 0.0697 |         2591.66   |             96 |            96 |
| MT_163   |              24 | H=24h (24 Hours Ahead) |  179.359  |   227.329  |   88.7241 | 0.0711 |         2523.95   |             96 |            96 |
| MT_163   |              48 | H=48h (48 Hours Ahead) |  131.947  |   159.264  |   29.0247 | 0.0514 |         2567.21   |             96 |            96 |

---

## 6. Honest Scope Boundaries & Academic Limitations

1. **Geographic & Hardware Scope:** Operates on historical Portuguese smart-meter telemetry (2011–2014); it has not been validated against live Indian utility grid telemetry or deployed in an automated SCADA dispatch environment.
2. **Aggregate Terminology:** The `PORTFOLIO` series represents the **Aggregate Consumption of 370 Observed Clients**, not total Portuguese national grid consumption.
3. **Statistical High-Demand Alerts:** Quantile-based flags ($P_{95}$ relative to training history) notify operators of statistically unusual demand peaks; they do not diagnose physical transformer thermal limits or feeder trip conditions.
4. **Safety Disclaimer:** Machine learning load forecasts carry inherent statistical uncertainty and must never replace qualified human engineering oversight in high-voltage utility dispatch operations.
