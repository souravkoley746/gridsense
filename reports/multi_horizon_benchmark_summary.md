# Multi-Horizon Forecasting Evaluation & Error Degradation

**Model:** Selected Autoregressive AI Forecaster (Recursive Multi-Step Projection)  
**Horizons Evaluated:** H = 1h (Next-Hour), 6h, 12h, 24h (Day-Ahead), and 48h (Two Days Ahead)  
**Sampling Protocol:** Stratified rolling-origin backtesting across all 24 hours of the day (4 origins per hour of day = 96 origins per client) to eliminate diurnal trough sampling bias.

---

## 1. Macro-Average Error Degradation Across Representative Clients

| Horizon   | Description     |   Macro-Average MAE (kWh) |   Macro-Average RMSE (kWh) |   Macro-Average ME (kWh) |   Normalized MAE (NMAE) |   Mean Actual (kWh) |   Evaluated Pairs |
|:----------|:----------------|--------------------------:|---------------------------:|-------------------------:|------------------------:|--------------------:|------------------:|
| H = 1h    | Next-Hour Ahead |                   315.31  |                    489.509 |                  91.9122 |                  0.0998 |             6461.61 |               960 |
| H = 6h    | 6 Hours Ahead   |                   392.542 |                    579.124 |                  67.0931 |                  0.1594 |             6534.33 |               960 |
| H = 12h   | 12 Hours Ahead  |                   407.264 |                    573.426 |                   4.5137 |                  0.1858 |             6594.47 |               960 |
| H = 24h   | 24 Hours Ahead  |                   478.126 |                    683.892 |                  -4.6974 |                  0.1681 |             6626.51 |               960 |
| H = 48h   | 48 Hours Ahead  |                   644.073 |                    944.46  |                 212.281  |                  0.3041 |             6522.26 |               960 |

---

## 2. H=1 Reconciliation (Stratified Rolling Origins vs Full Test Partition)

| client   | selected_model   | evaluation_column_used   |   stratified_origins_count |   rolling_origin_h1_mae_kwh |   full_test_partition_h1_mae_kwh |   test_total_hours |
|:---------|:-----------------|:-------------------------|---------------------------:|----------------------------:|---------------------------------:|-------------------:|
| MT_001   | XGBoost          | XGBoost                  |                         96 |                      0.9941 |                           0.927  |               3920 |
| MT_141   | XGBoost          | XGBoost                  |                         96 |                      0.7217 |                           0.857  |               3920 |
| MT_154   | XGBoost          | XGBoost                  |                         96 |                      1.441  |                           1.3507 |               3920 |
| MT_311   | XGBoost          | XGBoost                  |                         96 |                      6.6984 |                           5.7545 |               5234 |
| MT_343   | XGBoost          | XGBoost                  |                         96 |                     12.5459 |                           9.8241 |               3920 |
| MT_008   | XGBoost          | XGBoost                  |                         96 |                     13.5435 |                          12.2191 |               3920 |
| MT_362   | XGBoost          | XGBoost                  |                         96 |                   2953.76   |                        2803.83   |               3920 |
| MT_341   | Random Forest    | Random Forest            |                         96 |                     42.6637 |                          43.239  |               3920 |
| MT_131   | Random Forest    | Random Forest            |                         96 |                     19.3453 |                          14.1615 |               5234 |
| MT_163   | Random Forest    | Random Forest            |                         96 |                    101.386  |                          95.4021 |               5234 |

> [!NOTE]
> **Methodology & Error Growth:**  
> 1. Next-hour (H=1h) achieves the lowest error due to direct visibility of recent historical lags.  
> 2. As horizon extends to 6h, 12h, 24h, and 48h, errors compound through recursive autoregression without teacher forcing.  
> 3. Normalized MAE ($NMAE = MAE / \mu$) tracks percentage degradation relative to actual load level, ensuring bulk substation scale does not distort degradation trends.
