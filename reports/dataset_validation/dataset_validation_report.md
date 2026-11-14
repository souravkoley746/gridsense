# UCI ElectricityLoadDiagrams20112014: Validation Audit Report

**Audit Status:** `PASS`  
**Dataset DOI:** `10.24432/C58C86` (UCI ID: 321, CC BY 4.0)  
**Author:** Artur Trindade (INESC Porto / University of Porto, Portugal, 2015)  
**Verification Date:** 2026-11-14T14:59:07 UTC  

---

## 1. Cryptographic File Verification
- **Extracted Dataset File:** `LD2011_2014.txt` (678.06 MB)
- **SHA-256 Checksum:** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
- **Archive Checksum (Zip):** `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`

## 2. Dimensional & Temporal Integrity
- **Total Rows (Observations):** 140,256 (Expected: 140,256) -> **PASS**
- **Total Columns:** 371 (1 Timestamp + 370 Clients) -> **PASS**
- **Date Range:** `2011-01-01 00:15:00` to `2015-01-01 00:00:00` (4 complete calendar years)
- **Duplicate Timestamps:** 0 -> **PASS**
- **Monotonic Progression:** True -> **PASS**
- **Interval Consistency:** Exactly 15 minutes (0 off-grid intervals) -> **PASS**

## 3. Data Quality & Bounds
- **Missing Values (NaN / Null):** 0 -> **PASS (0 missing values)**
- **Negative Values:** 0 -> **PASS (no physical negative power readings)**
- **Clients Active from Day 1:** 158 / 370
- **Clients with Structural Leading Zeros:** 212 / 370

## 4. Daylight Saving Time (DST) Verification
European Daylight Saving Time adjustments match original UCI documentation:
- **March (23h day):** All 370 clients record four 0.0 kW intervals between 01:00 and 02:00.
- **October (25h day):** Hourly consumption between 01:00 and 02:00 contains the aggregate 2-hour sum.
