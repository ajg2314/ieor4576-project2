"""EDA Agent — exploratory financial data analysis on SEC filing data.

Architecture: data-store pattern (not prompt injection)

The full dataset is written to a JSON file before the agent runs.
Tools read from that file directly — the agent's prompt only contains
a compact ~200-char summary of what data is available.

This solves the context-overflow problem: 8000+ records used to be
serialised as ~120KB of JSON in the prompt, causing Gemini to make
zero tool calls. Now the prompt is tiny; all data lives on disk.

Two-stage output:
  Stage 1 — agent makes tool calls (stats, charts, python); results
             captured in side-channel observation store.
  Stage 2 — fresh tiny LLM call synthesises observations → EDAFindings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python, ARTIFACTS_DIR
from tools.visualizer import line_chart, bar_chart, waterfall_chart

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Data store ─────────────────────────────────────────────────────────────────
# Written by load_eda_data() before the agent runs; read by tool wrappers.

_EDA_DATA_FILE: str = str(ARTIFACTS_DIR / "eda_data.json")
_EDA_DATA_SUMMARY: str = ""

_eda_observations: list[dict] = []  # side-channel: captured tool results


def load_eda_data(records: list[dict]) -> str:
    """Write compact records to disk and build a summary for the EDA prompt.

    Returns the summary string to embed in the agent's input message.
    """
    global _EDA_DATA_FILE, _EDA_DATA_SUMMARY

    path = str(ARTIFACTS_DIR / "eda_data.json")
    with open(path, "w") as fh:
        json.dump(records, fh)
    _EDA_DATA_FILE = path

    tickers = sorted(set(r.get("ticker", "") for r in records if r.get("ticker")))
    metrics = sorted(set(r.get("metric", "") for r in records if r.get("metric")))
    years = sorted(set(r.get("fiscal_year", "") for r in records if r.get("fiscal_year")))

    _EDA_DATA_SUMMARY = (
        f"Available data ({len(records)} records, 10-K annual filings):\n"
        f"Companies ({len(tickers)}): {', '.join(tickers)}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Fiscal years: {years[0] if years else '?'} – {years[-1] if years else '?'}\n"
    )
    return _EDA_DATA_SUMMARY


def clear_eda_store() -> None:
    global _eda_observations
    _eda_observations = []


def get_eda_observations() -> list[dict]:
    return list(_eda_observations)


def _load_records() -> list[dict]:
    """Load the stored EDA records from disk."""
    try:
        with open(_EDA_DATA_FILE) as fh:
            return json.load(fh)
    except Exception:
        return []


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


# ── Prompt ────────────────────────────────────────────────────────────────────

EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent for a Sector Analyst system.

The financial data is already stored on disk. Your tools read it automatically —
you do NOT need to pass records to any tool. Just call the tools with metric
names, ticker symbols, and parameters.

TOOLS:

1. `stats_tool(metric, group_by)` — statistics for a metric (mean, median, growth).
   metric: one of the metric names shown in the data summary below.
   group_by: "ticker" (default) or "fiscal_year".

2. `filter_group_tool(metric, ticker, year, group_by, aggregate_fn)` — filter data.
   All parameters except metric are optional.

3. `run_python(code)` — execute pandas/numpy code.
   Data is pre-loaded: use `df` (all records as a DataFrame) or `records` (list of dicts).
   Available columns: ticker, company, fiscal_year, metric, value, value_billions.
   Print every computed number to stdout. Keep code focused and short.
   Example:
     rev = df[df.metric=='revenue'].pivot(index='fiscal_year', columns='ticker', values='value_billions')
     print(rev.to_string())

4. `create_chart(chart_type, series_data, title, subtitle, y_format, filename)` — charts.
   Call at least twice:
   a) Revenue trend: chart_type="line", y_format="billions"
      series_data = {ticker: {fiscal_year: value_billions, ...}, ...}
   b) Operating margin %: chart_type="line", y_format="pct"
      Compute margin = operating_income / revenue per ticker per year first (run_python),
      then pass those computed values to create_chart.

ANALYSIS CHECKLIST:
- Revenue levels, YoY growth %, CAGR across companies
- Operating margin %, gross margin % trends
- R&D as % of revenue
- Rank companies by growth or margin
- Identify the outlier vs sector peers
- Is the sector expanding or contracting?

Make all your tool calls. You do not need to write a final message.
"""


# ── Tool wrappers ─────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def stats_tool(metric: str, group_by: str = "ticker") -> dict:
    """Compute statistics (mean, median, growth rate) for a financial metric.

    metric: metric name (e.g. 'revenue', 'operating_income', 'gross_profit')
    group_by: 'ticker' or 'fiscal_year'
    """
    records = [r for r in _load_records() if r.get("metric") == metric]
    if not records:
        return {"error": f"No records found for metric='{metric}'"}
    result = compute_statistics(records, ["value_billions"], group_by=group_by)
    _eda_observations.append({
        "tool": "stats_tool",
        "description": f"Statistics for {metric} grouped by {group_by}",
        "value": str(result)[:500],
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def filter_group_tool(
    metric: str,
    ticker: str | None = None,
    year: str | None = None,
    group_by: str = "ticker",
    aggregate_fn: str = "mean",
) -> dict:
    """Filter financial data by metric/ticker/year and aggregate.

    metric: required — e.g. 'revenue'
    ticker: optional — filter to one company
    year: optional — filter to one fiscal year
    group_by: 'ticker' or 'fiscal_year'
    aggregate_fn: 'mean', 'sum', 'max', 'min'
    """
    records = [r for r in _load_records() if r.get("metric") == metric]
    if ticker:
        records = [r for r in records if r.get("ticker") == ticker]
    if year:
        records = [r for r in records if r.get("fiscal_year") == year]
    if not records:
        return {"error": f"No records for metric='{metric}' ticker='{ticker}' year='{year}'"}
    result = group_and_filter(
        records, "ticker",
        group_by=group_by,
        aggregate_column="value_billions",
        aggregate_fn=aggregate_fn,
    )
    _eda_observations.append({
        "tool": "filter_group_tool",
        "description": f"{metric} by {group_by}" + (f" for {ticker}" if ticker else ""),
        "value": str(result)[:500],
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """Execute pandas/numpy code. df and records are pre-loaded from disk.

    Available: df (DataFrame), records (list of dicts).
    Columns: ticker, company, fiscal_year, metric, value, value_billions.
    Print results to stdout. Keep code short and focused.
    """
    preamble = (
        f"import json, pandas as pd, numpy as np\n"
        f"with open({repr(_EDA_DATA_FILE)}) as _f:\n"
        f"    records = json.load(_f)\n"
        f"df = pd.DataFrame(records)\n"
    )
    result = execute_python(preamble + code)
    stdout = result.get("stdout", "").strip()
    _eda_observations.append({
        "tool": "run_python",
        "description": "Python computation",
        "value": stdout[:600] if stdout else result.get("stderr", "")[:300],
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def create_chart(
    chart_type: str,
    series_data: dict,
    title: str,
    subtitle: str = "",
    y_format: str = "billions",
    filename: str = "chart",
) -> str:
    """Generate a professional consulting-style chart and save it to disk.

    chart_type: "line" | "bar" | "waterfall"
    series_data: {series_name: {x_label: y_value}}
      e.g. {"AMZN": {"2020": 386.1, "2021": 469.8}, "UPS": {"2020": 84.6, ...}}
    y_format: "billions" | "pct" | "raw"
    Returns: relative path to saved PNG.
    """
    if chart_type == "line":
        path = line_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "bar":
        path = bar_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "waterfall":
        path = waterfall_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    else:
        return f"Unknown chart_type '{chart_type}'. Use 'line', 'bar', or 'waterfall'."

    _eda_observations.append({
        "tool": "create_chart",
        "description": title,
        "value": path,
        "artifact_path": path,
    })
    return path


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_eda_agent() -> Agent:
    return Agent(
        name="EDA",
        model=_make_model(),
        instructions=EDA_PROMPT,
        tools=[stats_tool, filter_group_tool, run_python, create_chart],
    )
