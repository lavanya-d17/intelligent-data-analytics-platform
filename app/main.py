"""Intelligent Data Analytics Platform - Streamlit app (Sem 7: Upload, Quality, Clean, Explore).

Run from the project root:   streamlit run app/main.py
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import core...` works when launched by Streamlit

import pandas as pd
import plotly.express as px
import streamlit as st

from core import eda
from core.clean import apply_step, replay
from core.ingest import IngestError, list_sheets, load_file
from core.profile import describe_step, profile, suggest_fixes

SAMPLE_PATH = ROOT / "data" / "samples" / "messy_sales.csv"
PAGES = ["1. Upload", "2. Quality Report", "3. Clean", "4. Explore"]

st.set_page_config(page_title="Intelligent Data Analytics Platform", layout="wide")


# ----------------------------------------------------------------- state helpers
def init_state():
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("filename", None)
    st.session_state.setdefault("steps", [])
    st.session_state.setdefault("last_upload_key", None)


def set_dataset(df: pd.DataFrame, name: str):
    st.session_state.raw_df = df
    st.session_state.filename = name
    st.session_state.steps = []


def current_df():
    raw = st.session_state.raw_df
    return None if raw is None else replay(raw, st.session_state.steps)


def require_data() -> bool:
    if st.session_state.raw_df is None:
        st.info("Upload a dataset first (page 1).")
        return False
    return True


# ----------------------------------------------------------------- callbacks
def cb_apply(step: dict):
    st.session_state.steps.append(step)


def cb_apply_all(steps: list):
    for s in steps:
        st.session_state.steps.append(s)


def cb_undo():
    if st.session_state.steps:
        st.session_state.steps.pop()


def cb_reset():
    st.session_state.steps = []


# ----------------------------------------------------------------- pages
def page_upload():
    st.header("Upload your data")
    st.write("Supported formats: CSV, TSV, TXT, Excel (.xlsx). Max 100 MB.")

    up = st.file_uploader("Choose a file", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if up is not None:
        data = up.getvalue()
        sheet, readable = None, True
        if up.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = list_sheets(data)
                if len(sheets) > 1:
                    sheet = st.selectbox("This workbook has several sheets. Pick one:", sheets)
            except IngestError as e:
                st.error(str(e))
                readable = False
        key = f"{up.name}|{sheet}"
        if readable and st.session_state.last_upload_key != key:
            try:
                label = up.name if sheet is None else f"{up.name} [{sheet}]"
                set_dataset(load_file(data, up.name, sheet=sheet), label)
                st.session_state.last_upload_key = key
                st.success(f"Loaded {label}")
            except IngestError as e:
                st.error(str(e))

    if st.button("Use the sample messy dataset"):
        set_dataset(load_file(io.BytesIO(SAMPLE_PATH.read_bytes()), SAMPLE_PATH.name), SAMPLE_PATH.name)

    df = st.session_state.raw_df
    if df is not None:
        st.subheader(f"Preview: {st.session_state.filename}")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        st.dataframe(df.head(50))
        st.caption("Column types as loaded (text columns may secretly be numbers or dates; "
                   "the Quality Report finds these).")
        st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}))


def render_profile(df: pd.DataFrame, title: str):
    prof = profile(df)
    ov = prof["overview"]
    st.subheader(title)
    c = st.columns(5)
    c[0].metric("Quality score", f"{ov['quality_score']}/100")
    c[1].metric("Rows", f"{ov['rows']:,}")
    c[2].metric("Duplicate rows", ov["duplicate_rows"])
    c[3].metric("Missing cells", f"{ov['missing_cells']:,} ({ov['missing_pct']}%)")
    c[4].metric("Columns with issues", ov["flagged_columns"])
    rows = []
    for name, i in prof["columns"].items():
        issues = []
        if i["kind"] == "id":
            issues.append("identifier (left out of stats and charts)")
        if i["kind"] == "numeric_text":
            issues.append("numbers stored as text")
        if i["kind"] == "datetime_text":
            issues.append("dates stored as text")
        if i["null_tokens"]:
            issues.append(f"{i['null_tokens']} placeholders (N/A, -, ...)")
        if i["variants"]:
            issues.append("inconsistent spellings")
        if i["kind"] == "boolean_text":
            issues.append("yes/no answers stored as text")
        if i["mixed"]:
            issues.append("numbers and words mixed together (review by hand)")
        if i["constant"]:
            issues.append("same value in every row")
        if i["outlier_pct"] > 1:
            issues.append(f"{i['outliers']} outliers")
        rows.append({"column": name, "type": i["kind"], "missing": i["missing"],
                     "missing %": i["missing_pct"], "unique": i["unique"],
                     "issues": ", ".join(issues) or "none"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    return prof


def page_quality():
    st.header("Data quality report")
    if not require_data():
        return
    render_profile(st.session_state.raw_df, "Original data")
    st.write("Go to **3. Clean** to review suggested fixes.")


def page_clean():
    st.header("Clean your data")
    if not require_data():
        return
    raw, df = st.session_state.raw_df, current_df()
    steps = st.session_state.steps

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Suggested fixes")
        suggestions = suggest_fixes(profile(df))
        if not suggestions:
            st.success("No more issues found. Your data looks clean.")
        else:
            st.caption("Nothing is changed until you click Apply. After applying, new suggestions "
                       "may appear (for example, filling missing values after converting a column).")
            st.button(f"Apply all {len(suggestions)} suggestions", on_click=cb_apply_all,
                      args=(suggestions,), type="primary")
            for idx, s in enumerate(suggestions):
                with st.container(border=True):
                    st.markdown(f"**{describe_step(s)}**")
                    st.caption(s["reason"])
                    st.button("Apply", key=f"apply_{len(steps)}_{idx}", on_click=cb_apply, args=(s,))

    with right:
        st.subheader("Pipeline log")
        if steps:
            for n, s in enumerate(steps, 1):
                st.write(f"{n}. {describe_step(s)}")
            b1, b2 = st.columns(2)
            b1.button("Undo last", on_click=cb_undo)
            b2.button("Reset all", on_click=cb_reset)
        else:
            st.write("No steps applied yet.")

        st.divider()
        st.write("**Replay a saved pipeline on this dataset**")
        log_file = st.file_uploader("Pipeline JSON", type=["json"], key="logfile")
        if log_file is not None and st.button("Replay uploaded pipeline"):
            try:
                loaded = json.load(log_file)
                replay(raw, loaded)  # dry run: raises if a column is missing
                st.session_state.steps = loaded
                st.rerun()
            except Exception as e:
                st.error(f"Could not replay this pipeline on this dataset: {e}")

    st.divider()
    st.subheader("Before vs after")
    ba, bb = st.columns(2)
    with ba:
        render_profile(raw, "Before")
    with bb:
        render_profile(df, "After")

    num_cols = [c for c in raw.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        col = st.selectbox("Compare a numeric column", num_cols)
        raw_num = pd.to_numeric(raw[col], errors="coerce")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(raw_num.dropna(), title=f"{col} (before)"))
        c2.plotly_chart(px.histogram(df[col].dropna(), title=f"{col} (after)"))

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("Download cleaned CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_data.csv", mime="text/csv")
    d2.download_button("Download pipeline (JSON)", json.dumps(steps, indent=2, default=str),
                       file_name="pipeline.json", mime="application/json")


def page_explore():
    st.header("Explore your data")
    if not require_data():
        return
    df = current_df()
    if st.session_state.steps:
        st.caption(f"Using cleaned data ({len(st.session_state.steps)} cleaning steps applied).")
    else:
        st.caption("Using the original data. Clean it first for better results.")

    st.subheader("Key insights")
    insights = eda.generate_insights(df)
    for line in insights or ["No notable patterns found."]:
        st.write(f"- {line}")
    for w in eda.guardrails(df):
        st.warning(w)

    st.subheader("Summary statistics")
    t1, t2 = st.tabs(["Numeric columns", "Categorical columns"])
    with t1:
        st.dataframe(eda.summary_numeric(df))
    with t2:
        st.dataframe(eda.summary_categorical(df), hide_index=True)

    heat = eda.correlation_heatmap(df)
    if heat is not None:
        st.subheader("Correlations")
        st.plotly_chart(heat)

    st.subheader("Build a chart")
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", cols)
    y = c2.selectbox("Y axis (optional)", ["(none)"] + cols)
    fig = eda.chart_for(df, x, None if y == "(none)" else y)
    if fig is None:
        st.info("No suitable chart for that combination. Try a different pair.")
    else:
        st.plotly_chart(fig)


# ----------------------------------------------------------------- main
init_state()
st.sidebar.title("Data Analytics Platform")
page = st.sidebar.radio("Workflow", PAGES)
if st.session_state.raw_df is not None:
    st.sidebar.success(f"Dataset: {st.session_state.filename}")
    st.sidebar.caption(f"{len(st.session_state.steps)} cleaning steps applied")

{"1. Upload": page_upload, "2. Quality Report": page_quality,
 "3. Clean": page_clean, "4. Explore": page_explore}[page]()
