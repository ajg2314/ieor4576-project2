"""EDA Agent — performs exploratory financial data analysis on SEC filing data.

Step 2: Explore. This agent:
- Computes financial metrics: YoY growth rates, operating margins, sector medians
- Groups and compares companies within a sector
- Writes and executes Python (pandas, matplotlib) at runtime for deeper analysis
- Generates comparison charts (revenue trends, margin evolution, sector scatter plots)
- Returns structured EDAFindings with a specific key insight

Grab bag: Code Execution, Data Visualization
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python
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

3. `run_python` — write and execute pandas/matplotlib code at runtime.
   You MUST call this at least twice to generate charts. Required charts:
   a) Revenue trend chart — line chart with one series per company, x=year, y=revenue in billions.
      Use a dark background (plt.style.use('dark_background')), distinct colors per company,
      clear labels, gridlines, and a legend. Title: "Revenue Trend".
   b) Margin comparison chart — grouped bar chart of operating margin % per company per year,
      OR a line chart of margin over time. Title: "Operating Margin %".
   Additional charts are encouraged (R&D spend, growth rates, etc.)
   Save every chart: plt.savefig(f'{ARTIFACTS_DIR}/chart_name.png', dpi=150, bbox_inches='tight')
   plt.close() after each save. Print computed values to stdout.

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
    {"tool_name": "...", "description": "...", "value": ..., "artifact_path": null}
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
    Execute Python code (pandas, numpy, matplotlib) in a sandboxed subprocess.

    The preamble already imports: pandas as pd, numpy as np, matplotlib, plt.
    ARTIFACTS_DIR is pre-defined — save charts like:
        plt.savefig(f'{ARTIFACTS_DIR}/revenue_trend.png', dpi=150, bbox_inches='tight')
    Print computed metrics to stdout for capture.
    """
    return execute_python(code)


def build_eda_agent() -> Agent:
    return Agent(
        name="EDA",
        model=_make_model(),
        instructions=EDA_PROMPT,
        tools=[stats_tool, filter_group_tool, run_python],
    )
