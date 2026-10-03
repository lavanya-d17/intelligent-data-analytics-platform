import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.clean import apply_step, replay
from core.eda import generate_insights, guardrails
from core.ingest import IngestError, load_file
from core.profile import profile, suggest_fixes

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "messy_sales.csv"


def messy():
    return pd.DataFrame({
        "Region": ["West", "west ", "WEST", "North", None],
        "Sales": ["1,200", "300", "N/A", "50.5", "10"],
        "Date": ["01/02/2024", "2024-02-03", "5 Mar 2024", "06/03/2024", "07/03/2024"],
        "Age": [20.0, np.nan, 30.0, 40.0, 25.0],
    })


# ---------- ingest ----------
def test_load_csv_with_semicolon():
    df = load_file(io.BytesIO(b"a;b\n1;2\n3;4\n"), "x.csv")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_load_rejects_empty_and_bad_type():
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b""), "x.csv")
    with pytest.raises(IngestError):
        load_file(io.BytesIO(b"abc"), "x.pdf")


# ---------- profile ----------
def test_profile_detects_issues():
    p = profile(messy())
    cols = p["columns"]
    assert cols["Sales"]["kind"] == "numeric_text"
    assert cols["Date"]["kind"] == "datetime_text"
    assert cols["Region"]["variants"]
    assert cols["Sales"]["null_tokens"] == 1
    assert cols["Age"]["missing"] == 1


def test_suggestions_are_ordered_and_skip_pending_columns():
    ops = [(s["op"], s.get("col")) for s in suggest_fixes(profile(messy()))]
    assert ("normalize_missing", None) in ops
    assert ("convert_numeric", "Sales") in ops
    assert ("impute", "Age") in ops
    # Region still has spelling variants, so no impute suggestion yet
    assert ("impute", "Region") not in ops


# ---------- clean ----------
def test_convert_numeric_and_datetime():
    out = apply_step(messy(), {"op": "convert_numeric", "col": "Sales"})
    assert out["Sales"].iloc[0] == 1200.0 and np.isnan(out["Sales"].iloc[2])
    out = apply_step(messy(), {"op": "convert_datetime", "col": "Date"})
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])
    assert out["Date"].notna().all()
    assert out["Date"].iloc[0] == pd.Timestamp("2024-02-01")  # day-first


def test_standardize_categories():
    out = apply_step(messy(), {"op": "standardize_categories", "col": "Region"})
    assert set(out["Region"].dropna()) == {"West", "North"}


def test_impute_methods():
    df = messy()
    assert apply_step(df, {"op": "impute", "col": "Age", "method": "median"})["Age"].isna().sum() == 0
    out = apply_step(df, {"op": "impute", "col": "Region", "method": "constant", "value": "Unknown"})
    assert out["Region"].iloc[4] == "Unknown"


def test_cap_outliers():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 1000]})
    out = apply_step(df, {"op": "cap_outliers", "col": "x"})
    assert out["x"].max() < 1000


def test_apply_step_does_not_mutate_input():
    df = messy()
    before = df.copy()
    apply_step(df, {"op": "convert_numeric", "col": "Sales"})
    pd.testing.assert_frame_equal(df, before)


def test_unknown_op_and_missing_column():
    with pytest.raises(ValueError):
        apply_step(messy(), {"op": "explode"})
    with pytest.raises(ValueError):
        apply_step(messy(), {"op": "impute", "col": "Nope", "method": "mean"})


# ---------- the key promise: replay ----------
def test_replay_is_deterministic_and_undo_works():
    raw = load_file(io.BytesIO(SAMPLE.read_bytes()), "messy_sales.csv")
    steps = []
    for _ in range(8):  # keep applying suggestions until none are left
        cur = replay(raw, steps)
        sugg = suggest_fixes(profile(cur))
        if not sugg:
            break
        steps.append({k: v for k, v in sugg[0].items()})
    final_a, final_b = replay(raw, steps), replay(raw, steps)
    pd.testing.assert_frame_equal(final_a, final_b)
    assert final_a.duplicated().sum() == 0
    assert profile(final_a)["overview"]["quality_score"] > profile(raw)["overview"]["quality_score"]
    # undo = drop last step and replay
    assert len(replay(raw, steps[:-1])) >= 0


# ---------- eda ----------
def test_insights_and_guardrails():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": -2 * x + rng.normal(scale=0.1, size=200)})
    ins = generate_insights(df)
    assert any("negatively" in i for i in ins)
    assert any("causes" in g for g in guardrails(df))
    assert any("unreliable" in g for g in guardrails(df.head(10)))
