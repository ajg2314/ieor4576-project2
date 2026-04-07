"""EDA Agent — performs exploratory financial data analysis on SEC filing data.

Step 2: Explore. This agent:
- Computes financial metrics: YoY growth rates, operating margins, sector medians
- Groups and compares companies within a sector
- Writes and executes Python (pandas) at runtime for derived metrics
- Generates professional consulting-quality charts via create_chart
- Returns structured EDAFindings with a specific key insight

Grab bag: Code Execution, Data Visualization
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python
from tools.visualizer import line_chart, bar_chart, waterfall_chart
from models.schemas import EDAFindings

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent for a Sector Analyst system.

You receive a DataBundle containing SEC EDGAR financial data for one or more companies.
Your job is to explore the data and surface SPECIFIC findings — numbers, percentages,
trends, and anomalies — that will ground the hypothesis.

Do NOT summarize raw data. Compute metrics and find patterns.

TOOLS AVAILABLE:

1. `stats_tool` — compute descriptive statistics over financial records.
   Use this to get means, medians, growth rates, and correlations across companies.

2. `filter_group_tool` — segment data by company, year, or metric.
   Use to compare groups (e.g., "which companies have margin > 30%?")

3. `run_python` — write and execute pandas/numpy code for derived metrics.
   Use for margin calculations, CAGR, growth rates, ranking tables.
   Print computed values to stdout. Do NOT use matplotlib here — use create_chart instead.

4. `create_chart` — generate professional consulting-style charts (Bloomberg/McKinsey quality).
   You MUST call this at least twice. Required charts:
   a) Revenue trend line chart — one series per company, x=fiscal year, y=revenue in billions.
   b) Operating margin comparison — one series per company, operating_income/revenue as %.

   create_chart signature:
     chart_type: "line" | "bar" | "waterfall"
     series_data: dict mapping series_name → dict of {x_label: y_value}
       Example: {"AAPL": {"2019": 260.17, "2020": 274.52}, "MSFT": {"2019": 125.84, "2020": 143.02}}
     title: bold chart title string
     subtitle: gray subtitle (e.g., "Annual 10-K filings | USD billions")
     y_format: "billions" | "pct" | "raw"
     filename: short snake_case base name (e.g., "revenue_trend")

   IMPORTANT — building series_data from records:
   - Filter records where metric == "revenue" (or "operating_income", etc.)
   - Group by ticker, then by fiscal_year → value_billions
   - Only include 10-K rows (form == "10-K") for clean annual data
   - If you need margin %, compute it in run_python first, then pass computed values to create_chart

FINANCIAL ANALYSIS CHECKLIST:
- Revenue: absolute levels, YoY growth %, CAGR
- Profitability: gross margin %, operating margin %, net margin %
- Efficiency: R&D as % of revenue (for tech/semis)
- Cross-company: rank companies by growth rate, margin, or scale
- Trend: is the sector expanding or contracting? Accelerating or decelerating?
- Anomaly: which company is an outlier vs sector peers?

OUTPUT:
Return EDAFindings with:
- A list of specific findings (each with a tool name, description, and computed value)
- `key_insight`: the single most important pattern (e.g. "NVDA operating margin expanded
  from 20% to 55% over 4 years, far outpacing AMD at 8% and INTC at -4%")
- `recommended_hypothesis_direction`: what the Hypothesis agent should focus on

OUTPUT FORMAT: Your final response must be a single JSON object with these exact keys:
{
  "findings": [
    {"tool_name": "...", "description": "...", "value": ..., "artifact_path": "artifacts/chart_abc123.png or null"}
  ],
  "key_insight": "<the single most important pattern found, with specific numbers>",
  "recommended_hypothesis_direction": "<what the hypothesis agent should focus on>"
}
IMPORTANT: After all tool calls are complete, you MUST send one final text message
containing ONLY the JSON object above. Do not stop after the last tool call.
"""


@function_tool(strict_mode=False)
def stats_tool(records: list[dict], numeric_columns: list[str], group_by: str | None = None) -> dict:
    """Compute descriptive statistics (mean, median, std, correlations, growth rate) over financial records."""
    return compute_statistics(records, numeric_columns, group_by=group_by)


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
    """Filter financial records and optionally group/aggregate by a column (company, year, etc.)."""
    return group_and_filter(
        records, filter_column, filter_value=filter_value,
        filter_gt=filter_gt, filter_lt=filter_lt,
        group_by=group_by, aggregate_column=aggregate_column,
        aggregate_fn=aggregate_fn,
    )


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """
    Execute Python code (pandas, numpy) in a sandboxed subprocess.

    Use for computing derived metrics: margins, CAGR, growth rates, ranking tables.
    Print computed values to stdout for capture. Do NOT generate charts here —
    use create_chart instead for all visualizations.
    """
    return execute_python(code)


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
    Generate a professional consulting-style chart (Bloomberg/McKinsey quality).

    Args:
        chart_type: "line" | "bar" | "waterfall"
        series_data: For line/bar — {series_name: {x_label: y_value}}.
                     For waterfall — {label: value} (ordered dict, first/last are totals).
                     Example: {"AAPL": {"2019": 260.17, "2020": 274.52}, "MSFT": {"2019": 125.84}}
        title: Bold chart title shown above the chart.
        subtitle: Smaller gray subtitle (e.g. "Annual 10-K | USD billions").
        y_format: "billions" (formats as $260B/$1.2T), "pct" (formats as 23.4%), "raw" (no formatting).
        filename: Short snake_case base name (uuid suffix added). E.g. "revenue_trend".

    Returns:
        Relative path to saved PNG, e.g. "artifacts/revenue_trend_abc123.png"
    """
    if chart_type == "line":
        return line_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "bar":
        return bar_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "waterfall":
        return waterfall_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    else:
        return f"Unknown chart_type '{chart_type}'. Use 'line', 'bar', or 'waterfall'."


def build_eda_agent() -> Agent:
    return Agent(
        name="EDA",
        model=_make_model(),
        instructions=EDA_PROMPT,
        tools=[stats_tool, filter_group_tool, run_python, create_chart],
    )
