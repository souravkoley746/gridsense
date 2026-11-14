"""Download and extract the genuine UCI ElectricityLoadDiagrams20112014 dataset.

Dataset Information:
- UCI Dataset ID: 321
- DOI: 10.24432/C58C86
- Creator: Artur Trindade (2015)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Official URLs:
  1. https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip (Direct dataset archive, ~140 MB)
  2. https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip (UCI static bundle, ~249 MB)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "raw" / "real"
TARGET_TXT = REAL_DIR / "LD2011_2014.txt"
TARGET_ZIP = REAL_DIR / "LD2011_2014.txt.zip"
MANIFEST_PATH = ROOT / "data" / "dataset_manifest.json"

URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_curl(url: str, dest_path: Path) -> bool:
    print(f"Downloading via curl from: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    
    cmd = [
        "curl.exe",
        "-L",
        "--retry", "3",
        "--retry-delay", "2",
        "--connect-timeout", "20",
        "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GridSense/2.0 Research",
        "-o", str(tmp_path),
        url
    ]
    try:
        t0 = time.time()
        res = subprocess.run(cmd, check=True)
        if tmp_path.exists() and tmp_path.stat().st_size > 1024:
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.rename(dest_path)
            print(f"Downloaded successfully: {dest_path} ({dest_path.stat().st_size / (1 << 20):.2f} MB in {time.time() - t0:.1f}s)")
            return True
    except Exception as exc:
        print(f"Curl download failed for {url}: {exc}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return False


def extract_dataset(zip_path: Path, dest_dir: Path) -> Path:
    print(f"Verifying zip archive: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File {zip_path} is not a valid zip archive.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupted = zf.testzip()
        if corrupted:
            raise ValueError(f"Zip archive contains corrupted member: {corrupted}")
        
        namelist = zf.namelist()
        candidates = [m for m in namelist if m.endswith("LD2011_2014.txt") and not m.startswith("__MACOSX")]
        if not candidates:
            candidates = [m for m in namelist if m.lower().endswith(".txt") and not m.startswith("__MACOSX")]
        if not candidates:
            raise FileNotFoundError(f"No LD2011_2014.txt found in zip. Archive contains: {namelist}")

        target_member = candidates[0]
        extracted_path = dest_dir / "LD2011_2014.txt"
        print(f"Extracting '{target_member}' ({zf.getinfo(target_member).file_size / (1 << 20):.1f} MB) -> {extracted_path}...")
        
        with zf.open(target_member) as source, open(extracted_path, "wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)

    print(f"Extraction complete: {extracted_path} ({extracted_path.stat().st_size / (1 << 20):.1f} MB)")
    return extracted_path


def main():
    parser = argparse.ArgumentParser(description="Download and verify genuine UCI ElectricityLoadDiagrams20112014 dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    args = parser.parse_args()

    REAL_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_TXT.exists() and not args.force:
        print(f"Genuine dataset text file already exists: {TARGET_TXT} ({TARGET_TXT.stat().st_size / (1 << 20):.1f} MB)")
    else:
        success = False
        for url in URLS:
            if download_with_curl(url, TARGET_ZIP):
                success = True
                break
        if not success:
            print("ERROR: Failed to download dataset from official UCI mirror URLs.", file=sys.stderr)
            sys.exit(1)

        extract_dataset(TARGET_ZIP, REAL_DIR)

    # Compute checksums
    print("Computing SHA-256 checksums...")
    txt_sha = sha256_file(TARGET_TXT)
    txt_size = TARGET_TXT.stat().st_size
    zip_sha = sha256_file(TARGET_ZIP) if TARGET_ZIP.exists() else None
    zip_size = TARGET_ZIP.stat().st_size if TARGET_ZIP.exists() else None

    manifest = {
        "dataset_name": "ElectricityLoadDiagrams20112014",
        "uci_id": 321,
        "doi": "10.24432/C58C86",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "creator": "Artur Trindade",
        "donation_date": "2015-03-12",
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_urls": URLS,
        "files": {
            "LD2011_2014.txt": {
                "relative_path": "data/raw/real/LD2011_2014.txt",
                "bytes": txt_size,
                "megabytes": round(txt_size / (1 << 20), 2),
                "sha256": txt_sha,
            },
            "LD2011_2014.txt.zip": {
                "relative_path": "data/raw/real/LD2011_2014.txt.zip",
                "bytes": zip_size,
                "megabytes": round(zip_size / (1 << 20), 2) if zip_size else None,
                "sha256": zip_sha,
            } if TARGET_ZIP.exists() else None,
        },
        "verified_by": "scripts/download_real_dataset.py",
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest generated at {MANIFEST_PATH}")
    print(f"  LD2011_2014.txt: {txt_size:,} bytes | SHA-256: {txt_sha}")
    if TARGET_ZIP.exists():
        print(f"  Zip Archive:     {zip_size:,} bytes | SHA-256: {zip_sha}")


if __name__ == "__main__":
    main()
