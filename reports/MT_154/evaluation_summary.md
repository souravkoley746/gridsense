# Model Evaluation Summary: MT_154

## 1. Chronological Partitioning
- **Training Period:** 2012-01-08 01:00:00 to 2014-02-08 07:00:00 (18,295 hours, 70%)
- **Validation Period:** 2014-02-08 08:00:00 to 2014-07-21 15:00:00 (3,920 hours, 15%)
- **Test Period:** 2014-07-21 16:00:00 to 2015-01-01 00:00:00 (3,921 hours, 15%)
- **Common Test Timestamps:** 3,920 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 1.7169 kWh (Best Baseline)
- **Daily seasonal-naive:** MAE = 1.9862 kWh
- **Weekly seasonal-naive:** MAE = 3.2640 kWh
- **Random Forest:** MAE = 1.3044 kWh
- **XGBoost:** MAE = 1.2596 kWh (Selected Model)

**Selected Model:** **XGBoost** achieved the lowest validation MAE (1.2596 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 1.7169 | 1.7843 | 2.5806 | +0.0063 | +0.00% | +0.00% | Best Baseline Reference |
| Daily seasonal-naive | baseline | 1.9862 | 1.9228 | 2.8371 | +0.0374 | -7.76% | -9.94% | Baseline Reference |
| Weekly seasonal-naive | baseline | 3.2640 | 2.9114 | 4.1040 | +0.2218 | -63.17% | -59.03% | Baseline Reference |
| Random Forest | ml | 1.3044 | 1.3859 | 1.7734 | -0.0070 | +22.33% | +31.28% | Candidate (Higher Validation MAE) |
| XGBoost | ml | 1.2596 | 1.3507 | 1.7601 | +0.0626 | +24.30% | +31.79% | Selected (Lowest Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **XGBoost** achieved a 24.30% MAE improvement over the best baseline (Persistence).
- **Calibration (ME):** Test Mean Error is +0.0626 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 3,920 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
