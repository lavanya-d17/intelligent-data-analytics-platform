# Intelligent Data Analytics Platform

Guided, explainable, reproducible data analysis for non-experts.
Sem 7 scope: **Upload -> Quality Report -> Clean (with replayable pipeline) -> Explore**.

## Setup (once)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/main.py
```

Click **Use the sample messy dataset** on page 1 to try it immediately.

## Test

```bash
python -m pytest -q
```

## Project layout

```
app/main.py        Streamlit UI (thin: no business logic)
core/ingest.py     load CSV/Excel with friendly errors
core/profile.py    quality profiling + fix suggestions (with reasons)
core/clean.py      cleaning operations + replayable pipeline (apply_step, replay)
core/eda.py        stats, plain-language insights, guardrails, charts
core/utils.py      shared helpers
db/                MySQL history (added later)
tests/             pytest unit tests
data/samples/      messy_sales.csv + generator script
```

## How the pipeline works

Every cleaning action is a small dict, e.g. `{"op": "impute", "col": "Age", "method": "median"}`.
The cleaned data is always `replay(raw_data, steps)`, so **undo** = drop the last step,
and a saved `pipeline.json` can be **replayed** on any new file with the same columns.
