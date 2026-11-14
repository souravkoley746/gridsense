# Model Evaluation Summary: MT_341

## 1. Chronological Partitioning
- **Training Period:** 2012-01-08 01:00:00 to 2014-02-08 07:00:00 (18,295 hours, 70%)
- **Validation Period:** 2014-02-08 08:00:00 to 2014-07-21 15:00:00 (3,920 hours, 15%)
- **Test Period:** 2014-07-21 16:00:00 to 2015-01-01 00:00:00 (3,921 hours, 15%)
- **Common Test Timestamps:** 3,920 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 110.7328 kWh
- **Daily seasonal-naive:** MAE = 48.1119 kWh (Best Baseline)
- **Weekly seasonal-naive:** MAE = 62.2333 kWh
- **Random Forest:** MAE = 39.9436 kWh (Selected Model)
- **XGBoost:** MAE = 40.3049 kWh

**Selected Model:** **Random Forest** achieved the lowest validation MAE (39.9436 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 110.7328 | 111.8377 | 194.0982 | +0.2953 | -96.40% | -82.88% | Baseline Reference |
| Daily seasonal-naive | baseline | 48.1119 | 56.9426 | 106.1316 | +1.7654 | +0.00% | +0.00% | Best Baseline Reference |
| Weekly seasonal-naive | baseline | 62.2333 | 69.4203 | 109.0435 | +12.5285 | -21.91% | -2.74% | Baseline Reference |
| Random Forest | ml | 39.9436 | 43.2390 | 69.6921 | +4.2378 | +24.07% | +34.33% | Selected (Lowest Validation MAE) |
| XGBoost | ml | 40.3049 | 45.4886 | 69.7395 | -6.2963 | +20.11% | +34.29% | Candidate (Higher Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **Random Forest** achieved a 24.07% MAE improvement over the best baseline (Daily seasonal-naive).
- **Calibration (ME):** Test Mean Error is +4.2378 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 3,920 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
