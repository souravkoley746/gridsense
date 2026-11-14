# GridSense: Dataset Provenance & Academic Citation Card

This document establishes the official provenance, citation metadata, data governance, and physical conversion principles for the primary dataset powering the **GridSense** energy forecasting system.

---

## 1. Official Dataset Provenance

| Attribute | Specification |
|---|---|
| **Official Dataset Name** | ElectricityLoadDiagrams20112014 |
| **Repository** | UC Irvine Machine Learning Repository |
| **UCI Dataset ID** | 321 |
| **Digital Object Identifier (DOI)** | [10.24432/C58C86](https://doi.org/10.24432/C58C86) |
| **Primary Creator** | Artur Trindade (INESC Porto / University of Porto) |
| **Donation Date** | March 12, 2015 |
| **Distribution License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Official Download Source** | `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip` |
| **Direct Archive URL** | `https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip` |
| **Geographic Context** | Portugal (Substations and consumer smart meters) |
| **Temporal Coverage** | 2011-01-01 00:15:00 to 2015-01-01 00:00:00 (4 complete calendar years) |
| **Sampling Interval** | 15 minutes (96 readings per calendar day) |
| **Total Timestamps** | 140,256 intervals |
| **Monitored Clients** | 370 client electricity series (`MT_001` to `MT_370`) |
| **Measured Physical Unit** | Average real power demand in kilowatts (kW) |

---

## 2. Academic Citation

When citing or presenting this dataset in research or academic submissions, use the following standard citation format:

```bibtex
@misc{trindade2015electricityload,
  author       = {Artur Trindade},
  title        = {{ElectricityLoadDiagrams20112014}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C58C86},
  note         = {{DOI: 10.24432/C58C86. Licensed under CC BY 4.0}}
}
```

**APA 7th Edition Citation:**
> Trindade, A. (2015). *ElectricityLoadDiagrams20112014* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86.

---

## 3. Comparison with Alternative Public Electricity Datasets

During the dataset evaluation phase, three prominent publicly available alternatives were benchmarked:

| Criterion | UCI ElectricityLoadDiagrams20112014 (Selected) | UCI Individual Household Electric Power (ID 235) | London DataStore SmartMeter Energy | Open Power System Data (OPSD Time-Series) |
|---|---|---|---|---|
| **Client Count** | **370 diverse series** | 1 single household | 5,567 households | 0 (National aggregate only) |
| **Granularity** | **15 minutes (96/day)** | 1 minute | 30 minutes | 15 / 60 minutes |
| **Missing Values** | **0 (Complete time grid)** | ~1.25% missing | Variable missing data | Variable missing data |
| **Suitability for Smart Grid AI** | **Ideal**: Evaluates individual client diversity and multi-client generalization | **Limited**: Single residence cannot evaluate multi-client portfolio dynamics | **Complex**: Highly fragmented multi-GB files with unaligned meter dates | **National Only**: Cannot evaluate smart meter load profiles |
| **License** | **CC BY 4.0** | CC BY 4.0 | UK Open Government Licence | CC BY 4.0 / ODC-BY |

**Selection Rationale:** The UCI ElectricityLoadDiagrams20112014 dataset provides an optimal balance of multi-client diversity (370 series), uniform 15-minute sampling over 4 years without random missing entries, and strict open licensing.

---

## 4. Physical Energy Conversion Rules

The raw dataset values represent **average power demand in kilowatts (kW)** over each 15-minute interval ($t_{15} = 0.25\text{ hours}$).

### Interval Energy Conversion:
$$\text{Energy per 15-minute interval (kWh)} = \text{Power (kW)} \times 0.25\text{ h}$$

### Hourly Energy Target Construction:
For each complete clock hour $h$ consisting of four consecutive 15-minute readings $P_{h,1}, P_{h,2}, P_{h,3}, P_{h,4}$:
$$\text{Hourly Energy } E_h\text{ (kWh)} = \sum_{j=1}^4 P_{h,j} \times 0.25\text{ h} = 0.25 \times (P_{h,1} + P_{h,2} + P_{h,3} + P_{h,4})$$

**Important Consistency Guarantee:** The model's target is consistently named and measured as **Hourly Electrical Energy in Kilowatt-Hours (kWh)**, labeled at the closing timestamp of the observation hour.

---

## 5. Daylight Saving Time (DST) Governance

In adherence to the official UCI repository documentation:
1. **March Time Change (Astronomical 23-Hour Day):**  
   On the last Sunday of March in Central European / Western European time, the clock advances by one hour at 01:00. The UCI dataset maintains exactly 96 intervals per day by recording **zero values (0.00 kW)** for the four 15-minute intervals between 01:00 and 02:00.
2. **October Time Change (Astronomical 25-Hour Day):**  
   On the last Sunday of October, the clock repeats the hour between 01:00 and 02:00. The UCI dataset records the **two-hour aggregate** within the 01:00–02:00 timestamp window.
3. **Pipeline Rule:** GridSense **does not** delete, duplicate, or alter timestamps. The complete 96-reading daily structure is preserved, and DST transition intervals are explicitly flagged (`dst_march`, `dst_october`) and isolated from fair scoreable evaluation without altering raw data.

---

## 6. Zero-Value Governance: Structural vs. Operational Zeros

- **Structural Leading Zeros:** Some clients were provisioned and integrated into the metering network after January 1, 2011. Readings prior to the client's first non-zero timestamp represent inactive pre-connection periods. The preprocessing pipeline automatically trims these leading structural zeros prior to feature engineering.
- **Legitimate Operational Zeros:** Zero readings occurring after a client's activation date reflect true periods of zero electrical consumption (e.g., disconnected load, vacancy, or zero active demand). These are strictly preserved and modeled.

---

## 7. Explicit Academic Boundaries

> [!WARNING]
> - **Historical Portuguese Context:** The data was gathered from electricity substations and smart meters in Portugal between 2011 and 2014. It **must not** be described as live electricity-grid telemetry, Indian smart-meter measurements, or real-time SCADA feeds.
> - **Academic Prototype Scope:** GridSense is a forecasting and visualization prototype. High-demand alerts represent statistical percentiles relative to training distributions ($P_{95}$), not physical circuit breaker or transformer limit breaches.
> - **Portfolio Labeling:** Any summation over all 370 client series must strictly be labeled as **"Aggregate Consumption of 370 Observed Clients"**, never as national grid demand.
