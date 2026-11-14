# GridSense: Data Quality & Preprocessing Assurance Report

**Execution Timestamp:** 2026-11-14T15:00:39 UTC  
**Dataset Source:** UCI *ElectricityLoadDiagrams20112014* (Artur Trindade, DOI: 10.24432/C58C86)  
**Conversion Formulation:** $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25\text{ kWh}$  

---

## 1. Quality Summary Table

| Target    |   Valid Hourly Records |   Scoreable Hours |   Trimmed Zero Hours |   DST Hours Flagged |   Mean kWh |   Std kWh |   Max kWh |
|:----------|-----------------------:|------------------:|---------------------:|--------------------:|-----------:|----------:|----------:|
| MT_001    |                 26,305 |            26,297 |                8,760 |                   6 |       5.29 |      6.04 |     34.58 |
| MT_141    |                 26,305 |            26,297 |                8,760 |                   6 |       8.84 |      4.63 |     56.21 |
| MT_154    |                 26,305 |            26,297 |                8,760 |                   6 |      33.24 |      5.17 |     73.13 |
| MT_311    |                 35,065 |            35,055 |                    0 |                   8 |     129.86 |     36.79 |    229.1  |
| MT_343    |                 26,305 |            26,297 |                8,760 |                   6 |     151.62 |    125.95 |    607.82 |
| MT_008    |                 26,305 |            26,297 |                8,760 |                   6 |     255.15 |     57.87 |    525.25 |
| MT_362    |                 26,305 |            26,297 |                8,760 |                   6 |   50139.4  |  36515.9  | 191000    |
| MT_341    |                 26,305 |            26,297 |                8,760 |                   6 |    1038.83 |    524.66 |   2187.66 |
| MT_131    |                 35,065 |            35,055 |                    0 |                   8 |      69.55 |     94.26 |    721.35 |
| MT_163    |                 35,065 |            35,055 |                    0 |                   8 |    2398.92 |    360.4  |   4592.17 |
| PORTFOLIO |                 35,065 |            35,055 |                    0 |                   8 |  195580    |  83888    | 448149    |

---

## 2. Preprocessing & Quality Safeguards
1. **Physical Energy Conservation:** 15-minute average power (kW) converted to hourly energy (kWh) via multiplication by 0.25h.
2. **Incomplete Interval Defense:** Any hourly window with fewer than 4 readings is flagged `incomplete=True` and excluded from model loss calculation and test scoring.
3. **European Daylight Saving Time (DST):** March 23-hour transitions and October 25-hour transitions are flagged and isolated from model benchmarks.
4. **Leading Structural Zero Handling:** Pre-activation structural zeros are trimmed on a per-client basis without altering the underlying raw dataset.
