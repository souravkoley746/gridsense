# Dataset Card — GridSense

## 1. Primary Dataset: UCI ElectricityLoadDiagrams20112014 (REAL)
* **Official Repository:** UCI Machine Learning Repository (ID: 321, DOI: [10.24432/C58C86](https://doi.org/10.24432/C58C86))
* **Donated by:** Artur Trindade (INESC Porto / University of Porto, 2015)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Expected File Location:** `data/raw/real/LD2011_2014.txt`
* **Automated Download Script:** `python scripts/download_real_data.py`
* **Archive SHA-256:** `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`
* **Extracted Text SHA-256:** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
* **Dimensions:** 140,256 rows × 371 columns (1 timestamp + 370 client series).
* **Format:** Semicolon (`;`) delimited, European decimal comma (`3,5`), double-quoted numeric values.
* **Granularity & Units:** 15-minute interval-end power measurements in kilowatts (kW), converted in GridSense to hourly energy in kilowatt-hours (kWh) via $E_h = \sum_{j=1}^4 P_{h,j} \times 0.25$.
* **Customer-Type Label Note:** The UCI dataset does **not** provide customer-type labels (residential, commercial, industrial). All 10 representative series (`MT_001`, `MT_141`, `MT_154`, `MT_311`, `MT_343`, `MT_008`, `MT_362`, `MT_341`, `MT_131`, `MT_163`) are characterized strictly by objective statistical profiles.
* **Decommissioned Series Audit:** Series `MT_223` was permanently decommissioned in late 2012 (70,080 consecutive zero readings in 2013–2014, 100% constant zero in test period). It was excluded and replaced by `MT_008` (100% active in test period).

---

## 2. Bundled Fast Demo Dataset: `data/raw/synthetic/LD2011_2014_demo.txt` (SYNTHETIC)
* **Status:** Demo / CI fallback only. Never used silently; requires explicit selection.
* **Generator:** `scripts/generate_demo_dataset.py` (seed 2024).
* **SHA-256:** `afd1e7ba95e84b3ca8193dd9a56d15d55b0c2d767a1b32c3619cefab7349882d`
* **Layout:** Identical syntax to UCI format (semicolon delimited, decimal comma, interval timestamps).
* **Scope:** 8 simulated series (`MT_001`…`MT_008`), 2012-01-01 to 2014-01-01 (70,176 rows).
* **Purpose:** Enables instant smoke tests in CI environments without downloading the full 711 MB UCI dataset.
