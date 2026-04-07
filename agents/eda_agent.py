"""EDA Agent — performs exploratory data analysis on collected data.

Step 2: Explore. This agent:
- Computes statistics and groups/filters data via deterministic tools
- Writes and executes Python code (pandas, matplotlib) at runtime
- Can fan out to multiple tools in parallel
- Returns structured EDAFindings with a key insight

Grab bag: Code Execution, Data Visualization
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python
from models.schemas import EDAFindings, DataBundle

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(
        model=LITELLM_MODEL_ID,
        extra_kwargs={
            "vertex_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "vertex_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        },
    )


EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent in a multi-agent data analysis system.

You receive a DataBundle from the Collector agent and the user's original question.
Your job is to explore the data thoroughly before forming any conclusions.

You must NOT summarize the raw data. You must compute specific metrics,
surface patterns, and identify anomalies that will ground the hypothesis.

Tools available:
1. stats_tool — compute means, medians, std, correlations, growth rates
2. filter_group_tool — segment data by category, time window, or threshold
3. run_python — write and execute pandas/numpy/matplotlib code at runtime.
   Use this for complex analysis that the other tools can't handle.
   For visualizations, save figures to artifacts/ using plt.savefig().

Process:
1. Examine the DataBundle. Identify numeric columns, categorical columns, time fields.
2. Run compute_statistics on key numeric columns.
3. If time series data: compute growth rates, rolling averages in Python code.
4. If categorical: use filter_group to segment and compare groups.
5. Generate at least one visualization using run_python (line chart, bar chart, etc.)
6. Identify the single most important pattern or anomaly you found.
7. Return structured EDAFindings with your findings and a recommended hypothesis direction.

Be specific. "Revenue grew 12% YoY" is better than "Revenue increased over time."
"""


@function_tool
def stats_tool(records: list[dict], numeric_columns: list[str], group_by: str | None = None) -> dict:
    """Compute descriptive statistics (mean, median, std, correlations, growth rate) over data records."""
    return compute_statistics(records, numeric_columns, group_by=group_by)


@function_tool
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
    """Filter records and optionally group and aggregate by a column."""
    return group_and_filter(
        records, filter_column, filter_value=filter_value,
        filter_gt=filter_gt, filter_lt=filter_lt,
        group_by=group_by, aggregate_column=aggregate_column,
        aggregate_fn=aggregate_fn,
    )


@function_tool
def run_python(code: str) -> dict:
    """
    Execute Python code (pandas, numpy, matplotlib) in a sandboxed subprocess.
    Save charts with: plt.savefig(f'{ARTIFACTS_DIR}/chart_name.png', dpi=150, bbox_inches='tight')
    Print computed values to stdout for capture.
    """
    return execute_python(code)


def build_eda_agent() -> Agent:
    return Agent(
        name="EDA",
        model=_make_model(),
        instructions=EDA_PROMPT,
        tools=[stats_tool, filter_group_tool, run_python],
        output_type=EDAFindings,
    )
