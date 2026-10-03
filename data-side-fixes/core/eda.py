"""Exploratory data analysis: stats, insights, guardrail warnings, charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.utils import is_id_like


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if not pd.api.types.is_bool_dtype(df[c]) and not is_id_like(df[c], c)]


def datetime_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="datetime").columns)


def categorical_cols(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    cols = []
    for c in df.columns:
        if c in numeric_cols(df) or c in datetime_cols(df) or is_id_like(df[c], c):
            continue
        if df[c].nunique(dropna=True) <= max_unique:
            cols.append(c)
    return cols


def summary_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_cols(df)
    return df[cols].describe().T.round(3) if cols else pd.DataFrame()


def summary_categorical(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in categorical_cols(df):
        s = df[c].dropna()
        if s.empty:
            continue
        top = s.value_counts().index[0]
        rows.append({"column": c, "unique": s.nunique(), "top_value": top,
                     "top_share_%": round(float((s == top).mean() * 100), 1)})
    return pd.DataFrame(rows)


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_cols(df)
    return df[cols].corr(numeric_only=True) if len(cols) >= 2 else pd.DataFrame()


def _strength_word(r: float) -> str:
    r = abs(r)
    return "very strongly" if r >= 0.8 else "strongly" if r >= 0.6 else "moderately"


def generate_insights(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Rule-based plain-language findings, strongest first."""
    found: list[tuple[float, str]] = []

    corr = correlation(df)
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.5:
                direction = "positively" if r > 0 else "negatively"
                found.append((abs(r), f"{a} and {b} are {_strength_word(r)} {direction} "
                                      f"correlated (r = {r:.2f})."))

    for c in numeric_cols(df):
        x = df[c].dropna()
        if len(x) > 8 and x.nunique() > 2:
            sk = x.skew()
            if abs(sk) > 1:
                found.append((0.5 + min(abs(sk), 3) / 10,
                              f"{c} is highly skewed (skew = {sk:.2f}); the median "
                              f"({x.median():,.2f}) describes a typical value better than the "
                              f"mean ({x.mean():,.2f})."))

    for c in categorical_cols(df):
        s = df[c].dropna()
        if len(s) and s.nunique() > 1:
            top = s.value_counts()
            share = top.iloc[0] / len(s)
            if share >= 0.5:
                found.append((share, f"{c}: '{top.index[0]}' makes up {share:.0%} of all rows."))

    for c in df.columns:
        m = df[c].isna().mean()
        if m >= 0.05:
            found.append((0.4 + m / 10, f"{c} is {m:.0%} missing."))

    found.sort(key=lambda t: t[0], reverse=True)
    return [text for _, text in found[:top_n]]


def guardrails(df: pd.DataFrame) -> list[str]:
    """Warnings that stop users drawing wrong conclusions."""
    w = []
    if len(df) < 30:
        w.append(f"Only {len(df)} rows: statistics from so few rows are unreliable.")
    if len(numeric_cols(df)) >= 2:
        w.append("Correlation shows that two columns move together, not that one causes the other.")
    heavy = [c for c in df.columns if df[c].isna().mean() > 0.3]
    if heavy:
        w.append(f"Columns with over 30% missing values ({', '.join(heavy)}) may give misleading results.")
    return w


def correlation_heatmap(df: pd.DataFrame):
    corr = correlation(df)
    if corr.empty:
        return None
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Correlation matrix")
    return fig


def chart_for(df: pd.DataFrame, x: str, y: str | None = None):
    """Pick a sensible chart from the column types."""
    num, dt = numeric_cols(df), datetime_cols(df)

    if is_id_like(df[x], x) or (y is not None and is_id_like(df[y], y)):
        return None  # charts of identifier columns carry no information
    if y is not None and x in num and y in dt:
        x, y = y, x  # date vs number: put the date on the x axis

    if y is None or y == x:
        if x in num:
            return px.histogram(df, x=x, marginal="box", title=f"Distribution of {x}")
        if x in dt:
            counts = df.set_index(x).resample("MS").size().reset_index(name="rows")
            return px.line(counts, x=x, y="rows", markers=True, title=f"Rows per month by {x}")
        vc = df[x].value_counts().head(20).reset_index()
        vc.columns = [x, "count"]
        return px.bar(vc, x=x, y="count", title=f"Counts of {x}")

    if x in num and y in num:
        fig = px.scatter(df, x=x, y=y, opacity=0.6, title=f"{y} vs {x}")
        d = df[[x, y]].dropna()
        if len(d) > 2 and d[x].nunique() > 1:
            m, b = np.polyfit(d[x], d[y], 1)
            xs = np.array([d[x].min(), d[x].max()])
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="trend",
                                     line=dict(color="red")))
        return fig
    if x in dt and y in num:
        d = df[[x, y]].dropna().sort_values(x)
        return px.line(d, x=x, y=y, title=f"{y} over time")
    if x not in num and x not in dt and y in num:
        return px.box(df, x=x, y=y, title=f"{y} by {x}")
    if y not in num and y not in dt and x in num:
        return px.box(df, x=y, y=x, title=f"{x} by {y}")
    return None
