"""EDA Agent — exploratory financial data analysis on SEC filing data.

Architecture: data-store pattern (not prompt injection)

The full dataset is written to a JSON file before the agent runs.
Tools read from that file directly — the agent's prompt only contains
a compact ~200-char summary of what data is available.

Key design decisions:
- load_eda_data() auto-generates the two standard charts (revenue trend +
  operating margin) before Stage 1 runs, guaranteeing ≥2 charts per report.
- plot_metric() / plot_margins() give the agent one-liner chart creation —
  no need to manually build series_data from raw records.
- run_python() pre-loads df and records so the agent can do arbitrary analysis.

Two-stage output:
  Stage 1 — agent makes tool calls; results captured in _eda_observations.
  Stage 2 — fresh tiny LLM call synthesises observations → EDAFindings JSON.
"""

from __future__ import annotations

import json
import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python, ARTIFACTS_DIR
from tools.visualizer import line_chart, bar_chart, waterfall_chart

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Data store ─────────────────────────────────────────────────────────────────

_EDA_DATA_FILE: str = str(ARTIFACTS_DIR / "eda_data.json")
_EDA_DATA_SUMMARY: str = ""
_eda_observations: list[dict] = []


def _load_records() -> list[dict]:
    try:
        with open(_EDA_DATA_FILE) as fh:
            return json.load(fh)
    except Exception:
        return []


def _build_series(records: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """Build series_data = {ticker: {fiscal_year: value_billions}} for a metric."""
    series: dict[str, dict[str, float]] = {}
    for r in records:
        if r.get("metric") != metric:
            continue
        ticker = r.get("ticker", "")
        year = str(r.get("fiscal_year", ""))
        val = r.get("value_billions")
        if ticker and year and val is not None:
            series.setdefault(ticker, {})[year] = float(val)
    return series


def _auto_generate_charts(records: list[dict]) -> list[dict]:
    """Pre-generate the two standard charts before the EDA agent runs.

    Returns a list of observation dicts to pre-populate the store.
    This guarantees at least 2 charts in every report regardless of
    what the agent does in Stage 1.
    """
    auto_obs: list[dict] = []

    # 1. Revenue trend
    rev_series = _build_series(records, "revenue")
    if rev_series:
        tickers = sorted(rev_series.keys())
        path = line_chart(
            {t: rev_series[t] for t in tickers},
            "Revenue Trend",
            subtitle="Annual 10-K filings | USD billions",
            y_format="billions",
            filename="revenue_trend",
        )
        auto_obs.append({
            "tool": "auto_chart",
            "description": "Revenue Trend",
            "value": path,
            "artifact_path": path,
        })

    # 2. Operating margin %
    rev = _build_series(records, "revenue")
    op = _build_series(records, "operating_income")
    if rev and op:
        margin_series: dict[str, dict[str, float]] = {}
        for ticker in set(rev) & set(op):
            for year in set(rev[ticker]) & set(op[ticker]):
                r_val = rev[ticker][year]
                o_val = op[ticker][year]
                if r_val and r_val != 0:
                    margin_series.setdefault(ticker, {})[year] = round(o_val / r_val * 100, 2)
        if margin_series:
            path = line_chart(
                margin_series,
                "Operating Margin %",
                subtitle="Operating income / Revenue | Annual 10-K",
                y_format="pct",
                filename="operating_margin",
            )
            auto_obs.append({
                "tool": "auto_chart",
                "description": "Operating Margin %",
                "value": path,
                "artifact_path": path,
            })

    # 3. Gross margin % (if available)
    gp = _build_series(records, "gross_profit")
    if rev and gp:
        gm_series: dict[str, dict[str, float]] = {}
        for ticker in set(rev) & set(gp):
            for year in set(rev[ticker]) & set(gp[ticker]):
                r_val = rev[ticker][year]
                g_val = gp[ticker][year]
                if r_val and r_val != 0:
                    gm_series.setdefault(ticker, {})[year] = round(g_val / r_val * 100, 2)
        if gm_series:
            path = line_chart(
                gm_series,
                "Gross Margin %",
                subtitle="Gross profit / Revenue | Annual 10-K",
                y_format="pct",
                filename="gross_margin",
            )
            auto_obs.append({
                "tool": "auto_chart",
                "description": "Gross Margin %",
                "value": path,
                "artifact_path": path,
            })

    return auto_obs


def load_eda_data(records: list[dict]) -> str:
    """Write compact records to disk, pre-generate standard charts, return summary."""
    global _EDA_DATA_FILE, _EDA_DATA_SUMMARY, _eda_observations

    path = str(ARTIFACTS_DIR / "eda_data.json")
    with open(path, "w") as fh:
        json.dump(records, fh)
    _EDA_DATA_FILE = path

    tickers = sorted(set(r.get("ticker", "") for r in records if r.get("ticker")))
    metrics = sorted(set(r.get("metric", "") for r in records if r.get("metric")))
    years = sorted(set(r.get("fiscal_year", "") for r in records if r.get("fiscal_year")))

    # Pre-generate standard charts and seed the observation store
    auto_obs = _auto_generate_charts(records)
    _eda_observations = list(auto_obs)  # reset store with auto charts

    _EDA_DATA_SUMMARY = (
        f"Available data ({len(records)} records, 10-K annual filings):\n"
        f"Companies ({len(tickers)}): {', '.join(tickers)}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Fiscal years: {years[0] if years else '?'} – {years[-1] if years else '?'}\n"
        f"Standard charts already generated: Revenue Trend, Operating Margin %, "
        f"Gross Margin % (if data available).\n"
        f"Use plot_metric() or plot_margins() to add more charts.\n"
    )
    return _EDA_DATA_SUMMARY


def clear_eda_store() -> None:
    """Clear observations BUT keep auto-generated charts (seeded by load_eda_data)."""
    # Do NOT clear — load_eda_data already reset the store with auto charts.
    # This function is a no-op; kept for API compatibility.
    pass


def get_eda_observations() -> list[dict]:
    return list(_eda_observations)


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


# ── Prompt ────────────────────────────────────────────────────────────────────

EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent for a Sector Analyst system.

The financial data is on disk. Your tools access it automatically — no need to
pass records as parameters. Standard charts (Revenue Trend, Operating Margin,
Gross Margin) are already generated. Your job is to add deeper analysis and
additional charts on top of those.

TOOLS:

1. `stats_tool(metric, group_by)` — statistics for one metric.
   metric: e.g. "revenue", "operating_income", "gross_profit", "rd_expense"
   group_by: "ticker" (default) or "fiscal_year"

2. `filter_group_tool(metric, ticker, year, group_by, aggregate_fn)` — filter data.

3. `run_python(code)` — pandas/numpy code. df and records are pre-loaded.
   df columns: ticker, company, fiscal_year, metric, value, value_billions
   Example:
     rev = df[df.metric=='revenue'].pivot(index='fiscal_year', columns='ticker', values='value_billions')
     growth = rev.pct_change() * 100
     print(growth.to_string())

4. `plot_metric(metric, chart_type, y_format, title)` — one-liner chart.
   Automatically groups by ticker and fiscal_year. Just name the metric.
   Examples:
     plot_metric("revenue")                          → revenue trend line chart
     plot_metric("rd_expense", y_format="billions")  → R&D spend chart
     plot_metric("net_income", title="Net Income Comparison")

5. `plot_margins(numerator_metric, denominator_metric, title)` — ratio/margin chart.
   Computes (numerator / denominator * 100) per company per year automatically.
   Examples:
     plot_margins("operating_income", "revenue", "Operating Margin %")
     plot_margins("rd_expense", "revenue", "R&D Intensity %")
     plot_margins("gross_profit", "revenue", "Gross Margin %")

6. `create_chart(chart_type, series_data, title, subtitle, y_format, filename)` — custom chart.
   Use when you need a waterfall, bar chart, or custom series (e.g. YoY growth rates).

ANALYSIS TASKS (do as many as the data supports):
- Call stats_tool on revenue and operating_income to get growth rates per company
- Call run_python to compute CAGR, rank companies by growth
- Call plot_metric("rd_expense") if R&D data exists — R&D intensity is key for tech
- Call plot_margins("rd_expense", "revenue", "R&D Intensity %") for tech sectors
- Identify the #1 outlier company and explain why with numbers

Make all your tool calls. You do not need to write any final message.
"""


# ── Tool wrappers ─────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def stats_tool(metric: str, group_by: str = "ticker") -> dict:
    """Compute statistics (mean, median, growth rate) for a financial metric.
    metric: e.g. 'revenue', 'operating_income', 'gross_profit', 'rd_expense'
    group_by: 'ticker' or 'fiscal_year'
    """
    records = [r for r in _load_records() if r.get("metric") == metric]
    if not records:
        return {"error": f"No records for metric='{metric}'"}
    result = compute_statistics(records, ["value_billions"], group_by=group_by)
    _eda_observations.append({
        "tool": "stats_tool",
        "description": f"Statistics for {metric} by {group_by}",
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
    """Filter and aggregate financial data by metric, ticker, and/or year.
    metric: required. ticker/year: optional filters.
    """
    records = [r for r in _load_records() if r.get("metric") == metric]
    if ticker:
        records = [r for r in records if r.get("ticker") == ticker]
    if year:
        records = [r for r in records if r.get("fiscal_year") == year]
    if not records:
        return {"error": f"No records for metric='{metric}' ticker='{ticker}' year='{year}'"}
    result = group_and_filter(
        records, "ticker", group_by=group_by,
        aggregate_column="value_billions", aggregate_fn=aggregate_fn,
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
    df columns: ticker, company, fiscal_year, metric, value, value_billions
    Print all computed values to stdout.
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
def plot_metric(
    metric: str,
    chart_type: str = "line",
    y_format: str = "billions",
    title: str | None = None,
) -> str:
    """Plot a financial metric for all companies automatically.

    One-liner chart creation — no need to build series_data manually.
    metric: e.g. 'revenue', 'rd_expense', 'net_income', 'gross_profit'
    chart_type: 'line' (default) or 'bar'
    y_format: 'billions' (default) or 'pct' or 'raw'
    title: optional override (default: metric name formatted)

    Returns: relative path to saved PNG.
    """
    records = _load_records()
    series = _build_series(records, metric)
    if not series:
        return f"No data found for metric='{metric}'"
    chart_title = title or metric.replace("_", " ").title()
    subtitle = "Annual 10-K filings"
    if chart_type == "bar":
        path = bar_chart(series, chart_title, subtitle=subtitle, y_format=y_format, filename=metric)
    else:
        path = line_chart(series, chart_title, subtitle=subtitle, y_format=y_format, filename=metric)
    _eda_observations.append({
        "tool": "plot_metric",
        "description": chart_title,
        "value": path,
        "artifact_path": path,
    })
    return path


@function_tool(strict_mode=False)
def plot_margins(
    numerator_metric: str,
    denominator_metric: str,
    title: str | None = None,
) -> str:
    """Compute and plot a margin/ratio (numerator/denominator * 100) per company per year.

    One-liner for any ratio chart — handles the computation automatically.
    Examples:
      plot_margins('operating_income', 'revenue', 'Operating Margin %')
      plot_margins('rd_expense', 'revenue', 'R&D Intensity %')
      plot_margins('gross_profit', 'revenue', 'Gross Margin %')

    Returns: relative path to saved PNG.
    """
    records = _load_records()
    num_series = _build_series(records, numerator_metric)
    den_series = _build_series(records, denominator_metric)
    if not num_series or not den_series:
        return f"Missing data for {numerator_metric} or {denominator_metric}"

    margin_series: dict[str, dict[str, float]] = {}
    for ticker in set(num_series) & set(den_series):
        for year in set(num_series[ticker]) & set(den_series[ticker]):
            d = den_series[ticker][year]
            if d and d != 0:
                pct = round(num_series[ticker][year] / d * 100, 2)
                margin_series.setdefault(ticker, {})[year] = pct

    if not margin_series:
        return f"Could not compute margins — no overlapping data"

    chart_title = title or f"{numerator_metric.replace('_',' ').title()} / {denominator_metric.replace('_',' ').title()} %"
    path = line_chart(
        margin_series, chart_title,
        subtitle="Annual 10-K filings",
        y_format="pct",
        filename=f"{numerator_metric}_over_{denominator_metric}",
    )
    _eda_observations.append({
        "tool": "plot_margins",
        "description": chart_title,
        "value": path,
        "artifact_path": path,
    })
    return path


@function_tool(strict_mode=False)
def create_chart(
    chart_type: str,
    series_data: dict,
    title: str,
    subtitle: str = "",
    y_format: str = "billions",
    filename: str = "chart",
) -> str:
    """Generate a custom chart with manually specified series_data.
    Use for waterfall charts, bar charts, or custom computed series (e.g. YoY growth %).
    series_data: {series_name: {x_label: y_value}}
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
        tools=[stats_tool, filter_group_tool, run_python,
               plot_metric, plot_margins, create_chart],
    )
