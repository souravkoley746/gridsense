# Model Evaluation Summary: MT_311

## 1. Chronological Partitioning
- **Training Period:** 2011-01-08 01:00:00 to 2013-10-21 19:00:00 (24,427 hours, 70%)
- **Validation Period:** 2013-10-21 20:00:00 to 2014-05-27 21:00:00 (5,234 hours, 15%)
- **Test Period:** 2014-05-27 22:00:00 to 2015-01-01 00:00:00 (5,235 hours, 15%)
- **Common Test Timestamps:** 5,234 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 10.0430 kWh
- **Daily seasonal-naive:** MAE = 8.5609 kWh
- **Weekly seasonal-naive:** MAE = 7.9644 kWh (Best Baseline)
- **Random Forest:** MAE = 5.8204 kWh
- **XGBoost:** MAE = 5.6616 kWh (Selected Model)

**Selected Model:** **XGBoost** achieved the lowest validation MAE (5.6616 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 10.0430 | 10.7469 | 15.6958 | +0.0231 | -18.70% | -22.75% | Baseline Reference |
| Daily seasonal-naive | baseline | 8.5609 | 9.2815 | 14.0116 | +0.0627 | -2.51% | -9.58% | Baseline Reference |
| Weekly seasonal-naive | baseline | 7.9644 | 9.0540 | 12.7866 | +0.4011 | +0.00% | +0.00% | Best Baseline Reference |
| Random Forest | ml | 5.8204 | 5.5849 | 7.5523 | -0.0174 | +38.32% | +40.94% | Candidate (Higher Validation MAE) |
| XGBoost | ml | 5.6616 | 5.7545 | 7.6852 | -0.2135 | +36.44% | +39.90% | Selected (Lowest Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **XGBoost** achieved a 36.44% MAE improvement over the best baseline (Weekly seasonal-naive).
- **Calibration (ME):** Test Mean Error is -0.2135 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 5,234 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
