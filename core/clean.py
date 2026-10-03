"""Cleaning operations and the replayable pipeline log.

A pipeline is just a list of dicts, e.g. {"op": "impute", "col": "Age", "method": "median"}.
Because it is plain data, it can be saved as JSON, undone, and replayed on a new file.
"""
from __future__ import annotations

import pandas as pd

from core.utils import (
    null_token_mask,
    tidy_text,
    to_boolean_loose,
    to_datetime_loose,
    to_numeric_loose,
)

COLUMN_OPS = {"convert_numeric", "convert_datetime", "convert_boolean", "standardize_categories",
              "impute", "cap_outliers", "replace_values", "rename_column"}
ALL_OPS = COLUMN_OPS | {"drop_duplicates", "normalize_missing", "drop_column", "drop_rows_missing"}


def _require_col(df: pd.DataFrame, col: str | None) -> str:
    if col is None or col not in df.columns:
        raise ValueError(f"Column '{col}' not found in this dataset.")
    return col


def _pick_spelling(group: pd.Series) -> str:
    """Most common spelling; on ties prefer 'Title Case' over ALL CAPS or all lowercase."""
    counts = group.value_counts()
    return sorted(counts.index, key=lambda v: (-counts[v], not v.istitle(), v.isupper(), v.islower(), v))[0]


def apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Apply one step and return a new DataFrame (the input is never modified)."""
    op = step.get("op")
    if op not in ALL_OPS:
        raise ValueError(f"Unknown operation: {op!r}")
    out = df.copy()

    if op == "drop_duplicates":
        return out.drop_duplicates().reset_index(drop=True)

    if op == "normalize_missing":
        for c in out.columns:
            m = null_token_mask(out[c])
            if m.any():
                out[c] = out[c].astype(object).where(~m, None)
        return out

    if op == "drop_column":
        return out.drop(columns=[step["col"]], errors="ignore")

    if op == "drop_rows_missing":
        threshold = float(step.get("threshold", 0.5))
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than 0 and at most 1")
        keep = out.isna().mean(axis=1) <= threshold
        return out[keep].reset_index(drop=True)

    col = _require_col(out, step.get("col"))

    if op == "convert_numeric":
        out[col] = to_numeric_loose(out[col])

    elif op == "convert_boolean":
        out[col] = to_boolean_loose(out[col])

    elif op == "replace_values":
        mapping = step.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("replace_values needs a non-empty 'mapping' dictionary")
        out[col] = out[col].map(lambda v: mapping.get(v, v))

    elif op == "rename_column":
        new_name = str(step.get("new_name", "")).strip()
        if not new_name:
            raise ValueError("rename_column needs a 'new_name'")
        if new_name != col and new_name in out.columns:
            raise ValueError(f"A column called '{new_name}' already exists")
        out = out.rename(columns={col: new_name})

    elif op == "convert_datetime":
        out[col] = to_datetime_loose(out[col], dayfirst=step.get("dayfirst", True))

    elif op == "standardize_categories":
        s = out[col]
        tidy = s.map(tidy_text)
        key = tidy.map(lambda v: v.lower() if isinstance(v, str) else v)
        canon = tidy.groupby(key).agg(_pick_spelling)
        out[col] = key.map(canon)

    elif op == "impute":
        method = step.get("method", "median")
        s = out[col]
        if method in ("mean", "median"):
            fill = s.mean() if method == "mean" else s.median()
            known = s.dropna()
            if len(known) and pd.notna(fill) and (known.astype(float) % 1 == 0).all():
                fill = round(fill)  # whole-number column: fill with a whole number
        elif method == "mode":
            m = s.mode()
            fill = m.iat[0] if not m.empty else None
        elif method == "constant":
            fill = step.get("value")
        else:
            raise ValueError(f"Unknown impute method: {method!r}")
        if fill is not None and not (isinstance(fill, float) and pd.isna(fill)):
            out[col] = s.fillna(fill)

    elif op == "cap_outliers":
        x = out[col].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        out[col] = x.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return out


def replay(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    """Re-run a list of steps from scratch on `df`. Undo = replay(df, steps[:-1])."""
    out = df.copy()
    for step in steps:
        out = apply_step(out, step)
    return out
