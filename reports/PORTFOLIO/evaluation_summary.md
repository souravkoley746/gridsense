# Model Evaluation Summary: PORTFOLIO

## 1. Chronological Partitioning
- **Training Period:** 2011-01-08 01:00:00 to 2013-10-21 19:00:00 (24,427 hours, 70%)
- **Validation Period:** 2013-10-21 20:00:00 to 2014-05-27 21:00:00 (5,234 hours, 15%)
- **Test Period:** 2014-05-27 22:00:00 to 2015-01-01 00:00:00 (5,235 hours, 15%)
- **Common Test Timestamps:** 5,234 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 13433.7466 kWh
- **Daily seasonal-naive:** MAE = 6984.7527 kWh (Best Baseline)
- **Weekly seasonal-naive:** MAE = 8839.8887 kWh
- **Random Forest:** MAE = 4104.0894 kWh (Selected Model)
- **XGBoost:** MAE = 4213.7323 kWh

**Selected Model:** **Random Forest** achieved the lowest validation MAE (4104.0894 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 13433.7466 | 17292.5207 | 26622.8518 | +34.8494 | -95.63% | -84.74% | Baseline Reference |
| Daily seasonal-naive | baseline | 6984.7527 | 8839.5262 | 14411.2544 | +139.6430 | +0.00% | +0.00% | Best Baseline Reference |
| Weekly seasonal-naive | baseline | 8839.8887 | 14990.9932 | 23788.8965 | +694.7053 | -69.59% | -65.07% | Baseline Reference |
| Random Forest | ml | 4104.0894 | 4592.4888 | 6712.1414 | +84.7237 | +48.05% | +53.42% | Selected (Lowest Validation MAE) |
| XGBoost | ml | 4213.7323 | 4827.9072 | 6883.6919 | -259.4168 | +45.38% | +52.23% | Candidate (Higher Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **Random Forest** achieved a 48.05% MAE improvement over the best baseline (Daily seasonal-naive).
- **Calibration (ME):** Test Mean Error is +84.7237 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 5,234 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
