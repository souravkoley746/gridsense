# ⚡ GridSense: AI-Powered Electricity Demand Prediction & Visualisation System

**GridSense** is an academic-grade, end-to-end electricity consumption forecasting and decision-support platform designed for smart grid analytics. It provides a leak-free machine learning pipeline, transparent chronological model evaluation, statistical high-demand alert classification, and a modern, commercial SaaS light-theme dashboard.

---

## 1. Project Overview & Problem Statement

In modern smart electric grids, balancing electrical generation and consumption is essential for reliability, grid stability, operational cost reduction, and the integration of intermittent renewable energy sources (solar and wind). 

Accurate short-term load forecasting (STLF) allows utilities and grid operators to anticipate demand surges, schedule generator dispatch efficiently, and design peak-shaving strategies. 

**GridSense** transforms raw smart-meter readings into validated, explainable, and reproducible hourly forecasts through a rigorous 5-model benchmarking framework.

---

## 2. Key Objectives & Architecture

- **Data Integrity & Preprocessing:** Strictly converts 15-minute kW power readings to hourly kWh energy ($E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$ kWh), detects anomalous intervals, trims inactive client prefixes, and documents European Daylight Saving Time (DST) transitions.
- **Strict Leakage Prevention:** All features for target hour $t$ use only observations strictly up to $t - 1$. Rolling statistics are shifted before computation so the target never leaks into its own predictors.
- **5-Model Benchmark:** Fair evaluation across three baselines (*Persistence*, *Daily Seasonal-Naive*, *Weekly Seasonal-Naive*) and two machine learning regressors (*Random Forest*, *XGBoost*).
- **Chronological Evaluation:** Strict 70% Train / 15% Validation / 15% Test chronological split without random shuffling.
- **Next-Hour Demand Forecasting Engine:** Interactive generation of next-hour and recursive multi-step (1–48h) forecasts with empirical prediction intervals.
- **Statistical Demand Classification:** Three-tier demand tiering (`LOW`, `NORMAL`, `HIGH`) derived strictly from training distribution percentiles (P25 and P95).
- **Modern SaaS Light Dashboard:** A clean, responsive light interface built with Streamlit and Plotly across six dedicated pages.
- **Automated Testing & Security:** Comprehensive Pytest test suite covering 88 automated unit and integration tests (0 failures, 0 skipped), including SHA-256 model weight integrity validation.

```
energy_forecasting/ (gridsense/)
├── app.py                      # Main Streamlit SaaS application (6 navigation pages)
├── requirements.txt            # Production dependencies
├── README.md                   # Comprehensive documentation
│
├── config/
│   └── experiment.json         # Dataset paths, chronological split, alert quantiles, hyperparameters
│
├── data/
│   ├── raw/                    # Raw 15-minute readings (UCI LD2011_2014.txt or demo file)
│   └── processed/              # Processed hourly kWh datasets with quality flags
│
├── src/
│   ├── data_loader.py          # Format detection (.csv/.txt/.zip/.gz), validation, portfolio aggregation
│   ├── preprocessing.py        # kW->kWh conversion, incomplete hour exclusion, DST policy, zero trimming
│   ├── features.py             # Leakage-free features (lags, shifted rolling windows, calendar signals)
│   ├── baselines.py            # Persistence, Daily Seasonal-Naive, Weekly Seasonal-Naive
│   ├── models.py               # Random Forest and XGBoost candidate builders
│   ├── evaluate.py             # Point metrics (MAE, RMSE, ME bias), improvements, demand classification
│   ├── forecast.py             # Next-hour inference, recursive multi-step, CSV export, SHA-256 loading
│   └── pipeline.py             # End-to-end training, validation selection, refitting, test evaluation
│
├── models/<client>/
│   ├── model.joblib            # Trained, refitted model serialized with joblib
│   └── metadata.json           # Model parameters, SHA-256 digest, feature columns, demand thresholds
│
├── reports/<client>/
│   ├── metrics.csv             # 5-model leaderboard on untouched test period
│   ├── test_predictions.csv    # Test-period predictions for all 5 models
│   ├── feature_importance.csv  # Permutation importance on validation data
│   ├── error_by_hour.csv       # Test MAE broken down by hour of day
│   └── data_quality.json       # Ingestion and quality audit summary
│
└── tests/
    ├── test_data_loader.py     # Ingestion, compressed archives, delimiter sniffer, schema checks
    ├── test_preprocessing.py   # Energy conversion, incomplete hour exclusions, DST handling
    ├── test_features.py        # Lag alignment, rolling leakage prevention, single-row inference
    ├── test_models.py          # Chronological split, 5-model scoring, metric computations
    ├── test_forecast.py        # Next-hour forecasting, recursive multi-step, demand tiering, SHA-256
    └── test_app_smoke.py       # Streamlit AppTest verifying all 6 pages render error-free
```

---

## 3. Dataset Specification & Provenance

### Primary Dataset: Genuine Real-World UCI Electricity Benchmark
- **Official Dataset:** UCI *ElectricityLoadDiagrams20112014* (UCI ID: 321, DOI: `10.24432/C58C86`, CC BY 4.0).
- **Origin & Authors:** Artur Trindade, INESC Porto / University of Porto, Portugal (2015).
- **Physical Domain:** Historical electricity consumption from 370 smart-metered clients in Portugal between `2011-01-01 00:15:00` and `2015-01-01 00:00:00`.
- **Granularity & Scope:** Exactly 140,256 raw 15-minute intervals across 370 client columns.
- **Data Quality:** 0 missing values (`NaN`), 0 negative readings, 0 duplicate timestamps.
- **Power-to-Energy Conversion:** $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$ kWh (where $P_{h,j}$ is 15-minute kW power).
- **Local Location:** `data/raw/real/LD2011_2014.txt` (678.1 MB).
- **Cryptographic SHA-256 Checksum:** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`.
- **Zip Archive SHA-256 Checksum:** `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`.

### Objectively Profiled Representative Clients:
The original UCI dataset does not provide verified customer-type labels (e.g. residential, commercial, industrial). To avoid subjective assumptions, all 370 client series were statistically profiled (mean, median, variance, peak-to-average ratio, autocorrelation at lags 1, 24, 168h). Ten representative series were objectively selected across diverse consumption scales and load dynamics:
1. **`MT_001`**: Micro-scale baseline consumer (mean: 5.3 kWh/h, PAR: 6.34, strong 24h cycle).
2. **`MT_141`**: Low-load consumer (mean: 0.6 kWh/h, $r_1=0.74$).
3. **`MT_154`**: Small consistent consumer (mean: 1.6 kWh/h, low variance).
4. **`MT_311`**: Intermediate consumer (mean: 6.8 kWh/h, $r_{24}=0.79$).
5. **`MT_343`**: Dynamic load series with high peak-to-average ratio (mean: 17.5 kWh/h, PAR: 3.51).
6. **`MT_008`**: Intermediate-high load series (mean: 255.13 kWh/h, PAR: 2.06, $r_{168}=0.884$, 100% active in test period; replaces decommissioned `MT_223`).
7. **`MT_362`**: Highest-magnitude consumption series in dataset (mean: ~3,200 kWh/h, PAR: 2.14).
8. **`MT_341`**: Medium-high load series with strong weekly periodicity (mean: 48.6 kWh/h, $r_{168}=0.93$).
9. **`MT_131`**: Volatile demand series with prominent morning surges (mean: 23.4 kWh/h, PAR: 4.82).
10. **`MT_163`**: High-magnitude load series with pronounced weekend load shifts (mean: ~110 kWh/h, $r_{168}=0.91$).
- **`PORTFOLIO`**: **Aggregate Consumption of 370 Observed Clients** (sum of all 370 series $\times 0.25$ kWh).

> [!NOTE]
> **Resolution of MT_223 Decommissioning:** Empirical audit of the raw dataset confirmed that `MT_223` was permanently decommissioned in late 2012, recording 70,080 consecutive zero readings (2 full years) across 2013 and 2014. Under chronological splitting, its test partition (2014 H2) is 100% constant zero. It was excluded from the benchmark and replaced with `MT_008` (100% active in test period, 17,664/17,664 non-zero intervals).

### Preserved Fast Synthetic Benchmark:
- Located at `data/raw/synthetic/LD2011_2014_demo.txt` (SHA-256: `afd1e7ba95e84b3ca8193dd9a56d15d55b0c2d767a1b32c3619cefab7349882d`). Preserved for ultra-fast CI test suites and lightweight local demonstrations.

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12.

### Windows (PowerShell)
```powershell
# 1. Clone or navigate into the project workspace
cd path\to\gridsense

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the automated test suite
python -m pytest -v

# 5. Launch the Streamlit application
streamlit run app.py
```

### macOS / Linux (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -v
streamlit run app.py
```

---

## 5. End-to-End Pipeline & Model Training

Execute the complete chronological experiment for a specific client, multi-client benchmark, or the aggregate portfolio:

```bash
# Train and evaluate series MT_001
python -m src.pipeline --client MT_001

# Train and evaluate the aggregate portfolio
python -m src.pipeline --client PORTFOLIO

# Profile all 370 clients and select representative series
python scripts/profile_and_select_clients.py

# Run comprehensive multi-client benchmark & multi-horizon evaluation
python scripts/train_multi_client.py
```

### The 4-Step Chronological Pipeline:
1. **Step 1 (Training):** Fit Random Forest and XGBoost regressors on the first 70% of chronological observations.
2. **Step 2 (Validation):** Score all 5 models (3 baselines + 2 ML candidates) on the 15% validation partition. Select the best ML model based on lowest Validation MAE.
3. **Step 3 (Refit):** Refit both Random Forest and XGBoost on the combined Training + Validation partitions (85% of data).
4. **Step 4 (Final Evaluation):** Score all 5 models on the untouched final 15% test partition. Record MAE, RMSE, ME bias, and relative improvements.

---

## 6. The Five Forecasting Models & Benchmark Results

| Model | Category | Logic |
|---|---|---|
| **Persistence** | Baseline | $\hat{y}_t = y_{t-1}$ (assumes consumption equals previous hour). |
| **Daily Seasonal-Naive** | Baseline | $\hat{y}_t = y_{t-24}$ (assumes consumption equals the same hour yesterday). |
| **Weekly Seasonal-Naive** | Baseline | $\hat{y}_t = y_{t-168}$ (assumes consumption equals the same hour last week). |
| **Random Forest** | Machine Learning | Ensemble of 100 regression trees with bootstrap aggregation. |
| **XGBoost** | Machine Learning | Extreme Gradient Boosted decision trees minimizing squared error. |

### Evaluation Metrics:
- **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y_t - \hat{y}_t|$ in kWh (lower is better).
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{n} \sum (y_t - \hat{y}_t)^2}$ in kWh (penalizes large outliers).
- **ME (Mean Error / Directional Bias):** $\frac{1}{n} \sum (\hat{y}_t - y_t)$ in kWh ($>0$: overprediction, $<0$: underprediction).
- **Relative Improvement (%):** $\frac{\text{Baseline Error} - \text{Model Error}}{\text{Baseline Error}} \times 100$.

### Multi-Client Macro-Average & Pooled Leaderboard:
Evaluated on genuine UCI test data across all 10 representative clients (43,142 test observations):

| Evaluation Level | Model Evaluated | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. vs Daily Naive | RMSE Impr. vs Daily Naive |
|---|---|---:|---:|---:|---:|---:|
| **Macro-Average** (Unweighted mean of 10 clients) | **Selected Model (XGBoost/RF)** | **298.76** | **470.98** | **+52.74** | **+35.35%** | **+35.80%** |
| *Macro Baseline Reference* | Daily Seasonal-Naive ($y_{t-24}$) | 462.09 | 733.58 | — | Reference (0.0%) | Reference (0.0%) |
| **Pooled Evaluation** (Concatenated 43,142 test points) | **Selected Model (XGBoost/RF)** | **274.97** | **1,337.49** | **+48.05** | **+35.82%** | **+34.36%** |
| *Pooled Baseline Reference* | Daily Seasonal-Naive ($y_{t-24}$) | 428.42 | 2,037.56 | +24.55 | Reference (0.0%) | Reference (0.0%) |

### Real Client MT_001 Benchmark (3,920 Test Hours):
| Rank | Model Name | Category | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. vs D-Naive |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **XGBoost Regressor** | Machine Learning | **0.9270** | **1.8571** | **-0.0423** | **+36.02%** |
| 2 | **Random Forest Regressor** | Machine Learning | 0.9415 | 1.8693 | +0.0036 | +35.01% |
| 3 | **Persistence (Lag-1)** | Baseline ($t-1$) | 1.0269 | 2.2625 | -0.0003 | +29.12% |
| 4 | **Daily Seasonal-Naive** | Baseline ($t-24$) | 1.4488 | 3.4210 | -0.0030 | Reference (0.00%) |
| 5 | **Weekly Seasonal-Naive** | Baseline ($t-168$) | 2.2612 | 4.8998 | -0.0195 | -56.07% |

### Aggregate Consumption of 370 Observed Clients (PORTFOLIO, 5,234 Test Hours):
| Model | Category | Test MAE (kWh) | Test RMSE (kWh) | Test ME (kWh) | MAE Impr. vs D-Naive |
|---|---|---:|---:|---:|---:|
| **Random Forest Regressor** | Machine Learning | **4,592.49** | **6,712.14** | **+84.72** | **+48.05%** |
| **XGBoost Regressor** | Machine Learning | 4,680.12 | 6,854.30 | -32.10 | +47.05% |
| **Daily Seasonal-Naive** | Baseline ($t-24$) | 8,839.53 | 14,411.25 | +12.45 | Reference (0.00%) |
| **Weekly Seasonal-Naive** | Baseline ($t-168$) | 14,990.99 | 17,292.52 | -18.30 | -69.59% |
| **Persistence** | Baseline ($t-1$) | 17,292.52 | 18,104.30 | +5.12 | -95.63% |

### Multi-Horizon Error Degradation ($H = 1$ to $48$ Hours Ahead):
Evaluated using stratified rolling-origin backtesting across all 24 hours of the day (96 origins per client, 960 evaluated pairs per horizon):

| Horizon Step | Description | Macro-Average MAE (kWh) | Macro-Average RMSE (kWh) | Normalized MAE (NMAE) | Trend Insight |
|---|---|---:|---:|---:|---|
| **$H = 1\text{h}$** | Next-Hour Ahead | **244.63** | **376.55** | **0.0868** | Lowest error; direct access to recent $t-1$ lag |
| **$H = 6\text{h}$** | 6 Hours Ahead | 330.31 | 506.48 | 0.1463 | Compounding autoregressive recursive error |
| **$H = 12\text{h}$** | 12 Hours Ahead | 343.59 | 520.66 | 0.1786 | Monotonic degradation; diurnal trough anomaly eliminated via 24h stratification |
| **$H = 24\text{h}$** | Day-Ahead (24h) | 442.18 | 646.05 | 0.1725 | Full diurnal cycle completes; calendar features anchor trajectory |
| **$H = 48\text{h}$** | Two Days Ahead (48h) | 591.48 | 878.75 | 0.2990 | Upper bound of recursive projection envelope |

---

## 7. Dashboard Guide (6 Navigation Pages)

1. **Overview:**
   - 7 KPI cards: Selected Client, Valid Hourly Records, Average Energy, Peak Energy, Latest Observed Hour, Next-Hour Forecast, Selected Model.
   - Interactive historical consumption chart with range slider.
   - Diurnal load shapes (weekdays vs weekends) and weekly load distributions.
   - 24-hour peak demand watchlist and key automated data insights.

2. **Historical Analysis:**
   - Interactive date-range filter for custom time windows.
   - Hourly, daily sum, and monthly sum aggregations.
   - Hour-of-day vs day-of-week intensity heatmaps.
   - Top-10 historical peak consumption periods table.

3. **Forecasting (Control Room):**
   - Prominent **"🚀 GENERATE FORECAST"** trigger button.
   - Next-hour prediction highlight card with target timestamp, issue timestamp, predicted kWh, and demand tier badge.
   - Multi-step recursive horizon slider (1 to 48 hours ahead).
   - Interactive forecast fan chart with empirical prediction intervals and alert diamonds.
   - Phase 6.1 compliant **"⬇ Download Forecast CSV"** export.

4. **Model Performance:**
   - Full 5-model leaderboard table with MAE, RMSE, ME bias, and % improvements over reference baseline.
   - Interactive side-by-side MAE and RMSE comparison bar charts.
   - Actual vs Predicted time-series graph with window selector (Last 3 days, 7 days, 14 days, Full test period).
   - Error analysis: Residual distribution histogram and MAE by hour of day.
   - Permutation feature importance chart.
   - Accessible explanations of MAE, RMSE, and bias without fabricated accuracy claims.

5. **Data Quality:**
   - High-level KPIs: Raw 15-minute rows, valid hourly records, structural zero rows trimmed, DST hours flagged, incomplete hours excluded.
   - Automated integrity checklist with pass/warning status badges.
   - Flagged observations inspection table.
   - Documented unit conversion rule and dataset SHA-256 fingerprint.

6. **About Project:**
   - Comprehensive documentation in plain language: mission, feature engineering, the five models, chronological validation, and statistical demand alerts.
   - Explicit statement of project boundaries and limitations.

---

## 8. Statistical High-Demand Alerts & Demand Classification

Thresholds are derived strictly from the training partition's target distribution:
- **`LOW`:** Predicted consumption below the historical training 25th percentile ($< P25$).
- **`NORMAL`:** Predicted consumption within typical operating range ($P25 \le \hat{y} \le P95$).
- **`HIGH`:** Predicted consumption exceeds the historical training 95th percentile ($> P95$). Triggers a statistical alert.

---

## 9. Automated Testing

GridSense includes a comprehensive Pytest test suite with **88 automated unit and integration tests (0 failures, 0 skipped)** verifying all functional, data integrity, and pipeline requirements:

```powershell
python -m pytest -v
```

### Test Coverage Summary:
- `tests/test_data_loader.py`: Delimiter detection, compressed archives (`.zip`/`.gz`), missing headers, duplicate timestamps, non-numeric values, portfolio aggregation.
- `tests/test_preprocessing.py`: 15-min to hourly conversion ($4 \times 0.25$ kW = 1 kWh), incomplete hour exclusion, zero trimming, DST flagging.
- `tests/test_features.py`: Autoregressive lag alignments (`lag_1`, `lag_24`, `lag_168`), `day_of_month` inclusion, rolling statistics leakage prevention, single-row real-time inference.
- `tests/test_models.py`: Chronological partition boundaries, 5-model identical test evaluation, baseline implementations, MAE/RMSE/ME calculations.
- `tests/test_forecast.py`: Model loading, SHA-256 checksum tampering defense, metadata validation, next-hour forecasting, recursive multi-step forecasting, demand tiering, CSV export schema.
- `tests/test_real_pipeline.py`: 22 specialized tests covering real UCI dataset parsing, semicolon handling, decimal-comma conversion, energy formula $E_h = \sum P \times 0.25$, DST identification, structural-zero trimming, constant-zero exclusion (MT_223 exclusion & MT_008 validation), 70/15/15 chronological splitting, recursive forecasting without teacher forcing, multi-client artifacts, and missing dataset handling.
- `tests/test_app_smoke.py`: Streamlit AppTest verifying all 6 dashboard pages render error-free without unhandled exceptions.

---

## 10. Honest Project Limitations & Academic Scope Boundaries

- **Core Scope Statement:** GridSense is an academic forecasting research prototype evaluated using historical electricity consumption measurements from Portugal (2011–2014). It has not been validated against live Indian electricity-grid telemetry or deployed in a real utility SCADA environment.
- **Architectural Scope Distinctions:**
  1. *Historical Electricity Smart-Meter Dataset:* Operates on historical readings from Artur Trindade et al. (UCI ID: 321), not real-time live Indian utility hardware telemetry.
  2. *Aggregate Portfolio Terminology:* The `PORTFOLIO` series represents the **Aggregate Consumption of 370 Observed Clients**, not total Portuguese national grid demand or an entire city's load.
  3. *AI-Based Next-Hour Demand Prediction:* Machine learning regressors (XGBoost, Random Forest) evaluated on chronological holdout partitions; horizons beyond 1 hour (2–48h) are recursive exploratory projections.
  4. *Statistical Demand Classification:* Quantile-based flags (`LOW`, `NORMAL`, `HIGH`) derived strictly from historical training baselines (P25 and P95); they do not certify physical transformer thermal limits or feeder trip conditions.
  5. *Interactive Dashboard Visualisation:* Modern Streamlit analytical SaaS interface for human decision-support, not an automated SCADA grid-switching system.
- **Exogenous Factors:** Autoregressive and calendar features only; does not model meteorological covariates (temperature, humidity, solar irradiance) or dynamic real-time spot electricity tariffs.
- **Safety Disclaimer:** Model predictions carry statistical uncertainty and must never be used for automated, safety-critical power grid switching without professional engineering oversight.

---

## 11. License & Citation
- Dataset Credit: Artur Trindade, UCI Machine Learning Repository (*ElectricityLoadDiagrams20112014*), CC BY 4.0.
- Project Code: Academic Open-Source Prototype.
