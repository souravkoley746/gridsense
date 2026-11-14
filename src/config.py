"""Configuration loading and project paths."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.json"


def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg


def dataset_path(cfg: dict) -> Path:
    p = Path(cfg["dataset"]["path"])
    full_p = p if p.is_absolute() else ROOT / p
    if full_p.exists():
        return full_p
    raise FileNotFoundError(
        f"Configured dataset was not found at '{full_p}'. "
        "GridSense strictly requires the genuine UCI ElectricityLoadDiagrams20112014 dataset. "
        "To download and verify the official dataset, please run: python scripts/download_real_data.py"
    )


def demo_dataset_path() -> Path:
    """Explicit path to synthetic demo dataset. Used ONLY when explicitly selected."""
    return ROOT / "data" / "raw" / "synthetic" / "LD2011_2014_demo.txt"


def safe_name(client: str) -> str:
    return "".join(ch for ch in client if ch.isalnum() or ch in "-_") or "series"


def model_dir(client: str) -> Path:
    return ROOT / "models" / safe_name(client)


def report_dir(client: str) -> Path:
    return ROOT / "reports" / safe_name(client)


def processed_path(client: str) -> Path:
    return ROOT / "data" / "processed" / f"{safe_name(client)}_hourly.csv"
