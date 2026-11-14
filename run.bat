@echo off
REM One-command launcher (Windows)
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
if not exist data\raw\real\LD2011_2014.txt (
    echo.
    echo [INFO] Genuine UCI ElectricityLoadDiagrams20112014 dataset not found at data\raw\real\LD2011_2014.txt.
    echo [INFO] GridSense will launch using bundled pre-trained models.
    echo [INFO] To download and train against the full 711MB real dataset, run: python scripts\download_real_data.py
    echo.
)
streamlit run app.py
