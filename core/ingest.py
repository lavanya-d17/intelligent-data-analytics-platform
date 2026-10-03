"""Loading files into DataFrames with friendly errors."""
from __future__ import annotations

import io

import pandas as pd

MAX_MB = 100


class IngestError(Exception):
    """Raised with a message that is safe to show directly to the user."""


def _read_text(raw: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            # sep=None lets pandas sniff the delimiter (comma, semicolon, tab ...)
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise IngestError(f"Could not decode the file: {last_err}")


def load_file(file, name: str | None = None) -> pd.DataFrame:
    """Load a CSV/TSV/TXT/Excel upload (file-like object or bytes)."""
    name = (name or getattr(file, "name", "") or "").lower()
    raw = file.read() if hasattr(file, "read") else file

    if not raw:
        raise IngestError("The file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise IngestError(f"The file is larger than {MAX_MB} MB.")

    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        elif name.endswith((".csv", ".tsv", ".txt")):
            df = _read_text(raw)
        else:
            raise IngestError("Unsupported file type. Please upload CSV, TSV, TXT or Excel.")
    except IngestError:
        raise
    except Exception as e:  # pandas parser errors, bad Excel files, ...
        raise IngestError(f"Could not read the file: {e}") from e

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise IngestError("The file has no rows or no columns.")

    df.columns = [str(c).strip() for c in df.columns]
    return df
