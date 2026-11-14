# Model Evaluation Summary: MT_008

## 1. Chronological Partitioning
- **Training Period:** 2012-01-08 01:00:00 to 2014-02-08 07:00:00 (18,295 hours, 70%)
- **Validation Period:** 2014-02-08 08:00:00 to 2014-07-21 15:00:00 (3,920 hours, 15%)
- **Test Period:** 2014-07-21 16:00:00 to 2015-01-01 00:00:00 (3,921 hours, 15%)
- **Common Test Timestamps:** 3,920 scoreable hourly observations evaluated across all 5 models

## 2. Validation Model Selection
Candidate forecasting models were trained strictly on the training partition and scored on the validation partition:
- **Persistence:** MAE = 23.2649 kWh
- **Daily seasonal-naive:** MAE = 21.3146 kWh
- **Weekly seasonal-naive:** MAE = 20.0094 kWh (Best Baseline)
- **Random Forest:** MAE = 12.3956 kWh
- **XGBoost:** MAE = 12.0808 kWh (Selected Model)

**Selected Model:** **XGBoost** achieved the lowest validation MAE (12.0808 kWh).
Both ML models were then refitted on combined training + validation data prior to test evaluation.

## 3. Test Evaluation Results (Untouched Test Period)
| Model Name | Model Type | Validation MAE (kWh) | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. (%) | RMSE Impr. (%) | Model-selection Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Persistence | baseline | 23.2649 | 23.8145 | 29.8678 | +0.0161 | -28.02% | -16.74% | Baseline Reference |
| Daily seasonal-naive | baseline | 21.3146 | 22.3244 | 32.0746 | -0.4003 | -20.01% | -25.36% | Baseline Reference |
| Weekly seasonal-naive | baseline | 20.0094 | 18.6018 | 25.5853 | -1.2957 | +0.00% | +0.00% | Best Baseline Reference |
| Random Forest | ml | 12.3956 | 12.5304 | 16.6281 | +0.1612 | +32.64% | +35.01% | Candidate (Higher Validation MAE) |
| XGBoost | ml | 12.0808 | 12.2191 | 16.0887 | +0.1438 | +34.31% | +37.12% | Selected (Lowest Validation MAE) |

## 4. Key Findings
- **Error Reduction:** Selected model **XGBoost** achieved a 34.31% MAE improvement over the best baseline (Weekly seasonal-naive).
- **Calibration (ME):** Test Mean Error is +0.1438 kWh, indicating near-zero average signed bias across the test period.
- **Fair Comparison:** All 5 models evaluated on the exact same 3,920 scoreable timestamps.

## 5. Dataset Provenance & Authenticity
- **Dataset Description:** Genuine UCI ElectricityLoadDiagrams20112014 dataset (UCI ID: 321, DOI: 10.24432/C58C86, CC BY 4.0).
- **Nature of Data:** Authentic electricity load measurements from 370 client series in Portugal (2011-2014) donated by Artur Trindade. 15-minute average power readings (kW) converted to hourly energy consumption (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
- **Dataset Checksum (SHA-256):** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **File Path:** `C:\Users\sourav\OneDrive\Desktop\gridsense\data\raw\real\LD2011_2014.txt`
- **Academic Scope:** GridSense is an academic forecasting prototype evaluated using historical electricity consumption data from Portugal. It has not been validated against live Indian electricity-grid measurements or deployed in a real smart-grid SCADA system.
