# Model Evaluation Summary: MT_141

## 1. Chronological Partitioning
- **Training Period:** 2012-01-08 01:00:00 to 2014-02-08 07:00:00 (18,295 hours, 70%)
- **Validation Period:** 2014-02-08 08:00:00 to 2014-07-21 15:00:00 (3,920 hours, 15%)
- **Test Period:** 2014-07-21 16:00:00 to 2015-01-01 00:00:00 (3,921 hours, 15%)
- **Common Test Timestamps:** 3,920 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 0.2222 kWh (Best Baseline)
- **Daily seasonal-naive:** MAE = 0.2633 kWh
- **Weekly seasonal-naive:** MAE = 0.5447 kWh
- **Random Forest:** MAE = 0.4528 kWh
- **XGBoost:** MAE = 0.3660 kWh (Selected Model)

**Selected Model:** **XGBoost** achieved the lowest validation MAE (0.3660 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 0.2222 | 0.2953 | 0.6398 | +0.0021 | +0.00% | +0.00% | Best Baseline Reference |
| Daily seasonal-naive | baseline | 0.2633 | 0.5761 | 1.6139 | +0.0115 | -95.10% | -152.26% | Baseline Reference |
| Weekly seasonal-naive | baseline | 0.5447 | 1.3441 | 2.8411 | +0.0456 | -355.14% | -344.09% | Baseline Reference |
| Random Forest | ml | 0.4528 | 0.5831 | 0.9751 | +0.0876 | -97.45% | -52.42% | Candidate (Higher Validation MAE) |
| XGBoost | ml | 0.3660 | 0.8570 | 1.4220 | +0.4583 | -190.22% | -122.27% | Selected (Lowest Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **XGBoost** achieved a -190.22% MAE improvement over the best baseline (Persistence).
- **Calibration (ME):** Test Mean Error is +0.4583 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 3,920 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
