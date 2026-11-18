"""Create the official GridSense_Final_Submission_FINAL.zip package for B.Tech Capstone delivery."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
OUTPUT_ZIP = DIST_DIR / "GridSense_Final_Submission_FINAL.zip"
MANIFEST_TXT = DIST_DIR / "SUBMISSION_MANIFEST.txt"
MANIFEST_JSON = DIST_DIR / "SUBMISSION_MANIFEST.json"

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    ".vercel",
    "gridsense_test_extracted",
    "gridsense_master_backup",
    "scratch",
    "data/raw/real",  # 678MB raw file downloaded on-demand via scripts/download_real_data.py
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".log",
}

EXCLUDE_FILES = {
    "GridSense_Final_Submission.zip",
    "GridSense_Final_Submission_FINAL.zip",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}


def should_exclude(rel_path: Path) -> bool:
    rel_str = rel_path.as_posix()
    for ex_dir in EXCLUDE_DIRS:
        if rel_str == ex_dir or rel_str.startswith(f"{ex_dir}/"):
            return True
    if rel_path.name in EXCLUDE_FILES:
        return True
    # Preserve official test execution audit log
    if rel_str == "reports/final_audit/test_execution.log":
        return False
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def calculate_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def make_zip():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Creating submission zip: {OUTPUT_ZIP}...")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    packed_entries = []
    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if not should_exclude(Path(root, d).relative_to(ROOT))]
            for file in sorted(files):
                abs_p = Path(root, file)
                rel_p = abs_p.relative_to(ROOT)
                if not should_exclude(rel_p):
                    archive_name = f"GridSense/{rel_p.as_posix()}"
                    zf.write(abs_p, archive_name)
                    f_size = abs_p.stat().st_size
                    f_hash = calculate_sha256(abs_p)
                    packed_entries.append({
                        "path": archive_name,
                        "size_bytes": f_size,
                        "sha256": f_hash,
                    })
                    file_count += 1
                    if file_count % 30 == 0:
                        print(f"  Added {file_count} files...")

    zip_size_bytes = OUTPUT_ZIP.stat().st_size
    zip_size_mb = zip_size_bytes / (1 << 20)
    zip_sha256 = calculate_sha256(OUTPUT_ZIP)

    # Generate Manifest
    manifest_lines = [
        "================================================================================",
        "GRIDSENSE B.TECH CAPSTONE FINAL SUBMISSION PACKAGE MANIFEST",
        "================================================================================",
        f"Archive Name:     {OUTPUT_ZIP.name}",
        f"Archive Location: {OUTPUT_ZIP.as_posix()}",
        f"Archive Size:     {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)",
        f"Archive SHA-256:  {zip_sha256}",
        f"Total Files:      {file_count}",
        "Compression:      ZIP_DEFLATED (level 6)",
        "Root Directory:   GridSense/",
        "================================================================================",
        "\nPACKED FILES INVENTORY:\n",
    ]
    for e in packed_entries:
        manifest_lines.append(f"{e['sha256']}  {e['size_bytes']:>10} B  {e['path']}")

    with open(MANIFEST_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest_lines) + "\n")
    with open(ROOT / "SUBMISSION_MANIFEST.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest_lines) + "\n")

    manifest_dict = {
        "archive_name": OUTPUT_ZIP.name,
        "archive_size_bytes": zip_size_bytes,
        "archive_size_mb": round(zip_size_mb, 2),
        "archive_sha256": zip_sha256,
        "total_files": file_count,
        "files": packed_entries,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as fh:
        json.dump(manifest_dict, fh, indent=2)

    print(f"\nPackaging complete!")
    print(f"  Archive:      {OUTPUT_ZIP}")
    print(f"  Total files:  {file_count}")
    print(f"  Archive size: {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"  SHA-256:      {zip_sha256}")
    print(f"  Manifest:     {MANIFEST_TXT}")
    return OUTPUT_ZIP, zip_sha256, file_count, zip_size_mb


if __name__ == "__main__":
    make_zip()
