"""Data ingestion, format detection, and schema validation (Phases 1.1-1.3).

Supports:
- Formats: .csv, .txt, .zip, .gz
- Delimiters: semicolon (;) and comma (,)
- Decimal formats: decimal comma (European UCI standard) and decimal period
- Validation: invalid schemas, missing timestamps, duplicate timestamps,
  non-numeric values, negative values, non-finite values, and interval checks.
"""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

PORTFOLIO = "PORTFOLIO"
DOCUMENTED_PORTFOLIO_CLIENTS = ["MT_001", "MT_002", "MT_003", "MT_004", "MT_005"]


class DataError(Exception):
    """Raised when the input dataset cannot be loaded or validated reliably."""


def sha256_file(path: Path | str) -> str:
    """Calculate the SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_stream(path: Path | str, member: str | None = None):
    """Return an open text stream, or file path/buffer suitable for pandas, and archive details."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"Dataset file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            namelist = [m for m in zf.namelist() if not m.startswith("__MACOSX") and not m.endswith("/")]
            if not namelist:
                raise DataError(f"Zip archive '{p.name}' contains no readable files.")
            if member:
                if member not in namelist:
                    raise DataError(f"Member '{member}' not found in zip archive. Available: {namelist}")
                chosen = member
            else:
                # Prioritize .txt or .csv files
                candidates = [m for m in namelist if m.lower().endswith((".txt", ".csv"))]
                chosen = candidates[0] if candidates else namelist[0]
            data_bytes = zf.read(chosen)
            return io.BytesIO(data_bytes), chosen, None
    elif suffix == ".gz":
        return p, None, "gzip"
    else:
        return p, None, None


def detect_dialect(path: Path | str, member: str | None = None) -> tuple[str, str]:
    """Detect delimiter (';' or ',') and decimal format (',' or '.') from the first lines."""
    stream_or_path, _, compression = _resolve_stream(path, member=member)
    sample_text = ""
    if isinstance(stream_or_path, io.BytesIO):
        stream_or_path.seek(0)
        sample_text = stream_or_path.read(4096).decode("utf-8", errors="replace")
        stream_or_path.seek(0)
    elif compression == "gzip":
        with gzip.open(stream_or_path, "rt", encoding="utf-8", errors="replace") as gz:
            sample_text = gz.read(4096)
    else:
        with open(stream_or_path, "r", encoding="utf-8", errors="replace") as fh:
            sample_text = fh.read(4096)

    lines = [ln.strip() for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        raise DataError("Dataset file is empty.")

    first_line = lines[0]
    semi_count = first_line.count(";")
    comma_count = first_line.count(",")
    sep = ";" if semi_count >= comma_count and semi_count > 0 else ","

    # Detect decimal comma by inspecting data lines
    decimal = "."
    if sep == ";":
        # In European semicolon files, comma is typically the decimal separator
        for line in lines[1:6]:
            if "," in line:
                decimal = ","
                break
    return sep, decimal


def _header(path: Path | str, member: str | None = None) -> tuple[list[str], str, str]:
    stream_or_path, _, compression = _resolve_stream(path, member=member)
    sep, decimal = detect_dialect(path, member=member)
    try:
        if isinstance(stream_or_path, io.BytesIO):
            stream_or_path.seek(0)
            df_hdr = pd.read_csv(stream_or_path, sep=sep, nrows=0)
        else:
            df_hdr = pd.read_csv(stream_or_path, sep=sep, compression=compression, nrows=0)
        cols = [str(c).strip().strip('"').strip("'") for c in df_hdr.columns.tolist()]
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"Could not read dataset header: {exc}") from exc

    if len(cols) < 2:
        raise DataError(
            f"Unsupported schema: expected a timestamp column and at least one client column. "
            f"Detected delimiter '{sep}', found {len(cols)} column(s)."
        )
    return cols, sep, decimal


def list_clients(path: Path | str, member: str | None = None) -> list[str]:
    """Client column names (excluding the timestamp column)."""
    cols, _, _ = _header(Path(path), member=member)
    return cols[1:]


def load_raw(
    path: Path | str,
    client: str,
    member: str | None = None,
    chunksize: int = 25000,
) -> tuple[pd.Series, dict]:
    """Load one client (or PORTFOLIO aggregation) as a kW series with 15-min timestamps."""
    path = Path(path)
    cols, sep, decimal = _header(path, member=member)
    ts_col_raw = cols[0]
    all_clients = cols[1:]

    # Handle portfolio vs individual client
    if client == PORTFOLIO:
        use = all_clients
        is_portfolio = True
        portfolio_desc = f"Aggregate Consumption of {len(all_clients)} Observed Clients"
    elif isinstance(client, list):
        use = [c for c in client if c in all_clients]
        if not use:
            raise DataError(f"None of the requested portfolio clients exist in dataset: {client}")
        is_portfolio = True
        portfolio_desc = f"Sum of {len(use)} selected clients ({', '.join(use[:5])}...)"
    elif client in all_clients:
        use = [client]
        is_portfolio = False
        portfolio_desc = None
    else:
        raise DataError(f"Client '{client}' not found. Available examples: {all_clients[:5]}")

    stream_or_path, member_name, compression = _resolve_stream(path, member=member)

    frames, nonnumeric = [], 0
    read_kwargs = {
        "sep": sep,
        "decimal": decimal,
        "chunksize": chunksize,
    }
    if compression:
        read_kwargs["compression"] = compression

    try:
        # Load chunks
        if isinstance(stream_or_path, io.BytesIO):
            stream_or_path.seek(0)
            reader = pd.read_csv(stream_or_path, **read_kwargs)
        else:
            reader = pd.read_csv(stream_or_path, **read_kwargs)

        for chunk in reader:
            # Clean headers
            chunk.columns = [str(c).strip().strip('"').strip("'") for c in chunk.columns]
            # Verify columns exist
            first_col = chunk.columns[0]
            chunk = chunk.rename(columns={first_col: "timestamp"})

            for c in use:
                if c not in chunk.columns:
                    raise DataError(f"Client column '{c}' missing in dataset chunk.")
                if not pd.api.types.is_numeric_dtype(chunk[c]):
                    # Parse numeric, replacing comma if needed
                    val_str = chunk[c].astype(str).str.strip().str.replace(",", ".", regex=False)
                    conv = pd.to_numeric(val_str, errors="coerce")
                    nonnumeric += int((chunk[c].notna() & conv.isna()).sum())
                    chunk[c] = conv
            frames.append(chunk[["timestamp"] + use])
    except Exception as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError(f"Failed to parse dataset observations: {exc}") from exc

    if not frames:
        raise DataError("Dataset contains no data rows.")

    df = pd.concat(frames, ignore_index=True)

    # Validate timestamps
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts = int(ts.isna().sum())
    if invalid_ts:
        raise DataError(f"Invalid timestamps detected ({invalid_ts} rows).")
    if nonnumeric:
        raise DataError(f"Non-numeric consumption values detected ({nonnumeric}).")
    if ts.duplicated().any():
        dup_count = int(ts.duplicated().sum())
        raise DataError(f"Duplicate timestamps detected ({dup_count}).")

    # Aggregate or select client values
    values = df[use].astype(float)
    kw = values.sum(axis=1, min_count=1) if is_portfolio else values.iloc[:, 0]
    series = pd.Series(kw.to_numpy(), index=pd.DatetimeIndex(ts), name=client if isinstance(client, str) else PORTFOLIO)

    was_unsorted = not series.index.is_monotonic_increasing
    if was_unsorted:
        series = series.sort_index()

    # Check non-finite
    if not np.isfinite(series.dropna().to_numpy()).all():
        raise DataError("Non-finite observations (NaN, Inf, or -Inf) detected.")

    negative_count = int((series < 0).sum())
    missing_count = int(series.isna().sum())
    valid_count = int(series.notna().sum() - negative_count)

    # Check 15-minute interval consistency
    diffs = series.index.to_series().diff().dropna()
    off_grid = int((diffs != pd.Timedelta(minutes=15)).sum())

    info = {
        "path": str(path),
        "archive_member": member_name,
        "sha256": sha256_file(path),
        "client": client if isinstance(client, str) else PORTFOLIO,
        "is_portfolio": is_portfolio,
        "portfolio_description": portfolio_desc,
        "clients_used": len(use),
        "raw_rows": int(len(series)),
        "valid_observations": valid_count,
        "raw_start": str(series.index.min()),
        "raw_end": str(series.index.max()),
        "was_unsorted": was_unsorted,
        "negative_values": negative_count,
        "missing_values": missing_count,
        "off_grid_intervals": off_grid,
        "source_unit": "kW",
        "delimiter": sep,
        "decimal_separator": decimal,
    }
    return series, info
