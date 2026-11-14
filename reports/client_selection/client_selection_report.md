# Representative Client Selection & Profiling Report

**Dataset Source:** UCI *ElectricityLoadDiagrams20112014* (Artur Trindade, DOI: 10.24432/C58C86)  
**Total Profiled Series:** 370 smart meters / substations  
**Selected Representative Series:** 10 diverse client series + 1 aggregate portfolio  

---

## 1. Selected 10 Representative Clients

| Client ID   | Active Start   |   Active Hours |   Mean (kWh/h) |   Median (kWh/h) |   Std Dev |   PAR |   CV |    r1 |   r24 |   r168 | Selection Rationale                                                                                                                        |
|:------------|:---------------|---------------:|---------------:|-----------------:|----------:|------:|-----:|------:|------:|-------:|:-------------------------------------------------------------------------------------------------------------------------------------------|
| MT_001      | 2012-01-01     |         26,305 |           5.29 |             2.22 |      6.04 |  6.53 | 1.14 | 0.867 | 0.754 |  0.585 | Preserved project baseline continuity: Micro-scale profile (mean 5.29 kWh/h) with strong 24h rhythm (r24=0.754).                           |
| MT_141      | 2012-01-01     |         26,305 |           8.83 |             7.44 |      4.63 |  6.36 | 0.52 | 0.982 | 0.966 |  0.914 | Lowest consumption tier: Minimum positive base load (mean 8.83 kWh/h, r1=0.982).                                                           |
| MT_154      | 2012-01-01     |         26,305 |          33.24 |            33.53 |      5.18 |  2.2  | 0.16 | 0.896 | 0.842 |  0.671 | Small stable load tier: Mean 33.24 kWh/h with low coefficient of variation (CV=0.16).                                                      |
| MT_311      | 2011-01-01     |         35,065 |         129.84 |           141.26 |     36.8  |  1.76 | 0.28 | 0.913 | 0.939 |  0.944 | Moderate load tier: Mean 129.84 kWh/h with strong diurnal regularity (r24=0.939).                                                          |
| MT_343      | 2012-01-01     |         26,305 |         151.59 |           126.69 |    125.95 |  4.01 | 0.83 | 0.925 | 0.977 |  0.95  | Dynamic load tier: Mean 151.59 kWh/h with high peak-to-average ratio (PAR=4.01).                                                           |
| MT_008      | 2012-01-01     |         26,305 |         255.13 |           253.37 |     57.88 |  2.06 | 0.23 | 0.862 | 0.849 |  0.884 | Intermediate-high load tier (Replacement for decommissioned MT_223): 100% active in test period (mean 255.13 kWh/h, PAR=2.06, r168=0.884). |
| MT_362      | 2012-01-01     |         26,305 |       50130.6  |         43500    |  36513.2  |  3.81 | 0.73 | 0.949 | 0.984 |  0.949 | Bulk substation load tier: Highest observed consumption scale in dataset (mean 50130.64 kWh/h).                                            |
| MT_341      | 2012-01-01     |         26,305 |        1038.64 |          1204.97 |    524.68 |  2.11 | 0.51 | 0.927 | 0.981 |  0.979 | Strong weekly cyclicity tier: High lag-168 autocorrelation (r168=0.979, mean 1038.64 kWh/h).                                               |
| MT_131      | 2011-01-01     |         35,065 |          69.56 |            33.85 |     94.26 | 10.37 | 1.36 | 0.826 | 0.527 |  0.735 | Volatile demand profile: High variance and morning peaks (CV=1.36, PAR=10.37).                                                             |
| MT_163      | 2011-01-01     |         35,065 |        2398.8  |          2413.59 |    360.59 |  1.91 | 0.15 | 0.904 | 0.503 |  0.755 | Heavy load tier: High-magnitude consumer (mean 2398.80 kWh/h, r168=0.755) with weekend drops.                                              |

---

## 2. Decommissioned Series Audit & Exclusion of MT_223

> [!IMPORTANT]
> **Resolution of MT_223 Statistical Contradiction:**  
> - **The Contradiction:** An earlier summary described MT_223 as having zero mean and zero variance, while the full 370-client profiling table listed a mean of 336.21 kW and 56.8% zero readings.
> - **The Physical Ground Truth:** Independent inspection of the raw data (`LD2011_2014.txt`) reveals that `MT_223` was fully active during 2011 (mean 804.08 kW) and 2012 (mean 540.23 kW), but was **permanently decommissioned in September 2012**.
> - **Test Partition Flaw:** Throughout all of 2013 and 2014 (70,080 consecutive 15-minute intervals), `MT_223` recorded exactly `0.0 kW`. Under chronological 70/15/15 splitting, its test partition (July–December 2014) is **100% constant zero**.
> - **Action Taken:** `MT_223` is formally excluded from the benchmarking suite because trailing structural zeros from decommissioned meters do not constitute a valid forecasting problem.
> - **Replacement:** **`MT_008`** was validated and selected as the replacement. `MT_008` is 100% active throughout the test period (17,664 out of 17,664 readings non-zero), with mean 256.14 kW, std 56.68 kW, and strong daily/weekly seasonality.
