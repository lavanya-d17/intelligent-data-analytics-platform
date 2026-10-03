"""Small helpers shared by profiling and cleaning."""
from __future__ import annotations

import re

import pandas as pd

# Strings that really mean "missing"
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}

DATE_RE = re.compile(
    r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"
)


def is_text_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def null_token_mask(s: pd.Series) -> pd.Series:
    """True where a value is a string like 'N/A', 'null', '-' or blank."""
    return s.map(lambda v: isinstance(v, str) and v.strip().lower() in NULL_TOKENS).astype(bool)


def to_numeric_loose(s: pd.Series) -> pd.Series:
    """'1,200.50', '$300', '12%' -> numbers. Unparseable values become NaN."""
    base = s.astype(object).where(~null_token_mask(s), None)

    def conv(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        if isinstance(v, (int, float)):
            return float(v)
        cleaned = re.sub(r"[,\s$₹€£%]", "", str(v))
        try:
            return float(cleaned)
        except ValueError:
            return float("nan")

    return base.map(conv).astype(float)


def to_datetime_loose(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    vals = s.astype(object).where(~null_token_mask(s), None)
    return pd.to_datetime(vals, errors="coerce", format="mixed", dayfirst=dayfirst)


def looks_numeric(s: pd.Series, threshold: float = 0.8) -> bool:
    """Text column where most values are really numbers (e.g. '1,200')."""
    if not is_text_dtype(s):
        return False
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return False
    return to_numeric_loose(vals).notna().mean() >= threshold


def looks_datetime(s: pd.Series, threshold: float = 0.8) -> bool:
    """Text column where most values are dates in some format."""
    if not is_text_dtype(s):
        return False
    vals = s.dropna()
    vals = vals[~null_token_mask(vals)]
    if vals.empty:
        return False
    matches = vals.map(lambda v: bool(DATE_RE.match(str(v)))).mean()
    if matches < threshold:
        return False
    return to_datetime_loose(vals).notna().mean() >= threshold


def tidy_text(v):
    """Trim and collapse inner whitespace; leave non-strings alone."""
    return re.sub(r"\s+", " ", v.strip()) if isinstance(v, str) else v
