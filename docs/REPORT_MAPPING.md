# Mapping to the project report

| Report item | Where it lives |
|---|---|
| FR01 load dataset / FR02 select client | `src/data_loader.py`, sidebar *Series* selector |
| FR03 validate | `src/data_loader.py`, `src/preprocessing.py`, *Data Quality* page |
| FR04 hourly observations (kWh) | `src/preprocessing.py::build_hourly` |
| FR05 features (no leakage) | `src/features.py` |
| FR06 baselines on same timestamps | `src/baselines.py`, `src/pipeline.py` |
| FR07 candidate models, seeds saved | `src/models.py`, `metadata.json` |
| FR08 MAE / RMSE / ME / improvement | `src/evaluate.py`, *Model Lab* |
| FR09 next-hour & multi-step forecast | `src/forecast.py`, *Forecast Studio* |
| FR10 high-consumption alert | threshold = train P95; *Overview*, *Forecast Studio*, *Model Lab › Alerts* |
| FR11 visualisation | `ui/charts.py`, `app.py` |
| FR12 export | *Forecast Studio* CSV, `reports/<client>/*.csv` |
| FR13 model metadata | `models/<client>/metadata.json` (+ SHA-256 integrity check) |
| Architecture / DFD / flowcharts | *Methodology* page, report Chapter 3 |
| Test cases TC01–TC15 | `tests/test_gridsense.py` (names carry the TC id) |
| Future scope delivered early | recursive multi-step forecast, empirical prediction intervals, permutation importance |

## Test-case status (fill the report's table from this)
Run `python -m unittest discover -s tests -v`. TC01–TC15 each have a named test. Replace “Not executed” in the report with the
actual result you observe on your machine.

## Things the report asked to be reported honestly
* Alerts are labelled *Forecast high-consumption alert* - never “grid overload”.
* No energy/cost/emission saving is claimed; the peak-shaving slider is arithmetic on the forecast only.
* A baseline is recommended automatically if it beats the ML models on validation.
