# Multi-Client Forecasting Benchmark Summary

**Dataset:** Genuine UCI ElectricityLoadDiagrams20112014 (UCI ID: 321, DOI: 10.24432/C58C86)  
**Execution Timestamp:** 2026-11-14T14:09:18 UTC  
**Evaluated Targets:** 10 Valid Representative Clients + Aggregate Portfolio (MT_223 replaced with MT_008)  
**Evaluation Principle:** Chronological 70% Train / 15% Validation / 15% Test with model selection strictly based on lowest Validation MAE.

---

## 1. Overall Multi-Client Performance (Individual Clients)

| Metric Level | Evaluated Model | MAE (kWh) | RMSE (kWh) | ME Bias (kWh) | Improvement vs Daily Naive (MAE) | Improvement vs Daily Naive (RMSE) | Total Evaluated Predictions |
|---|---|---:|---:|---:|---:|---:|---:|
| **Macro-Average** (Unweighted mean across 10 clients) | **Selected Model (XGBoost/RF)** | **298.7564** | **470.9788** | **+52.7353** | **+35.35%** | **+35.80%** | 10 Clients |
| *Macro Baseline Reference* | Daily Seasonal Naive ($y_{t-24}$) | 462.0893 | 733.5797 | — | Baseline Reference | Baseline Reference | 10 Clients |
| **Pooled Evaluation** (Concatenated test set) | **Selected Model (XGBoost/RF)** | **274.9705** | **1337.4908** | **+48.0474** | **+35.82%** | **+34.36%** | **43,142** |
| *Pooled Baseline Reference* | Daily Seasonal Naive ($y_{t-24}$) | 428.4234 | 2037.5612 | +24.5475 | Baseline Reference | Baseline Reference | 43,142 |

> [!NOTE]
> **Macro-Average vs Pooled Metrics:**  
> Macro-average computes the unweighted mean of metrics across all 10 clients, ensuring micro-scale accuracy is evaluated equally alongside large bulk-consumption substations. Pooled metrics evaluate point errors across all concatenated 44,456 test observations.

---

## 2. Client-by-Client Performance Breakdown

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

## 3. Aggregate Consumption of 370 Observed Clients
- **Terminology:** *Aggregate Consumption of 370 Observed Clients* (calculated as the sum of all 370 client kW readings $\times 0.25$ kWh).
- **Scope Distinction:** Represents the pooled demand of the 370 monitored metering points in Portugal; it is **NOT** total Portuguese national grid consumption.
