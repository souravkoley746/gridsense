#!/usr/bin/env bash
# One-command launcher (macOS / Linux): creates a venv, installs deps, starts the app.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f data/raw/real/LD2011_2014.txt ]; then
    echo ""
    echo "[INFO] Genuine UCI ElectricityLoadDiagrams20112014 dataset not found at data/raw/real/LD2011_2014.txt."
    echo "[INFO] GridSense will launch using bundled pre-trained models."
    echo "[INFO] To download and train against the full 711MB real dataset, run: python scripts/download_real_data.py"
    echo ""
fi
streamlit run app.py
