"""EDA Agent — performs exploratory financial data analysis on SEC filing data.

Two-stage design to avoid the context-overflow problem:

Stage 1 (tool-calling): The agent makes all its tool calls — stats, charts,
  Python execution. Results are captured in a side-channel store so they
  survive even if the agent never emits a final text message.

Stage 2 (synthesis): A fresh, clean LLM call receives only a compact summary
  of the tool results and produces the EDAFindings JSON. This call has <2KB of
  context and reliably produces output.
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python
from tools.visualizer import line_chart, bar_chart, waterfall_chart

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Side-channel store ─────────────────────────────────────────────────────────
# Tool wrappers append observations here so Stage 2 can synthesise findings
# even if the agent never emits a final text message.

_eda_observations: list[dict] = []  # {tool, description, value, artifact_path}


def clear_eda_store() -> None:
    global _eda_observations
    _eda_observations = []


def get_eda_observations() -> list[dict]:
    return list(_eda_observations)


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


# ── Prompt ────────────────────────────────────────────────────────────────────

EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent for a Sector Analyst system.

You receive annual 10-K financial data for a set of companies.
Your ONLY job right now is to call tools to explore the data.
Do NOT write any JSON or summary — a separate synthesis step handles that.

TOOLS:

1. `stats_tool` — descriptive statistics (mean, median, growth rate, correlations).
2. `filter_group_tool` — filter and group records by company, year, or metric.
3. `run_python` — pandas/numpy code for margins, CAGR, growth rates, rankings.
   Print every computed number to stdout. Keep code short and focused.
4. `create_chart` — professional consulting-style charts. Call at least twice:
   a) Revenue trend (line chart, y_format="billions")
   b) Operating margin % over time (line chart, y_format="pct")
   Additional charts encouraged: R&D spend, gross margin, revenue growth rate.

   create_chart(
     chart_type = "line" | "bar" | "waterfall",
     series_data = {ticker: {fiscal_year: value, ...}, ...},
     title = "...",
     subtitle = "Annual 10-K | ...",
     y_format = "billions" | "pct" | "raw",
     filename = "snake_case_name"
   )
   Build series_data from records: filter metric=="revenue", group by ticker→fiscal_year→value_billions.
   For margins: compute (operating_income / revenue) per company per year in run_python first.

ANALYSIS CHECKLIST — cover as many as the data supports:
- Revenue levels, YoY growth %, CAGR
- Gross margin %, operating margin %, net margin %
- R&D as % of revenue
- Cross-company ranking by growth or margin
- Sector expansion / contraction trend
- Outliers vs peers

Make all your tool calls. You do not need to write a final message.
"""


# ── Tool wrappers (also write to side-channel store) ─────────────────────────

@function_tool(strict_mode=False)
def stats_tool(records: list[dict], numeric_columns: list[str], group_by: str | None = None) -> dict:
    """Compute descriptive statistics (mean, median, std, growth rate) over financial records."""
    result = compute_statistics(records, numeric_columns, group_by=group_by)
    _eda_observations.append({
        "tool": "stats_tool",
        "description": f"Statistics on {numeric_columns} grouped by {group_by}",
        "value": str(result)[:400],
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def filter_group_tool(
    records: list[dict],
    filter_column: str,
    filter_value: str | None = None,
    filter_gt: float | None = None,
    filter_lt: float | None = None,
    group_by: str | None = None,
    aggregate_column: str | None = None,
    aggregate_fn: str = "mean",
) -> dict:
    """Filter financial records and optionally group/aggregate by a column."""
    result = group_and_filter(
        records, filter_column, filter_value=filter_value,
        filter_gt=filter_gt, filter_lt=filter_lt,
        group_by=group_by, aggregate_column=aggregate_column,
        aggregate_fn=aggregate_fn,
    )
    _eda_observations.append({
        "tool": "filter_group_tool",
        "description": f"Filter {filter_column}={filter_value or ''} gt={filter_gt} lt={filter_lt}, group={group_by}",
        "value": str(result)[:400],
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """Execute Python code (pandas, numpy) for derived metrics. Print results to stdout."""
    result = execute_python(code)
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
    """
    Generate a professional consulting-style chart.

    chart_type: "line" | "bar" | "waterfall"
    series_data: {series_name: {x_label: y_value}}
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
