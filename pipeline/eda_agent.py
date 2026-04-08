"""EDA Agent — exploratory financial data analysis on SEC filing data.

Data architecture:
  All records (10-K annual + 10-Q quarterly, full history) are loaded into
  a SQLite database at artifacts/eda.db before Stage 1 runs.

  The agent has four ways to access data:
    1. sql_query(sql)   — run any SELECT; returns rows as list of dicts
    2. run_python(code) — pandas + sklearn + scipy; df pre-loaded from DB
    3. plot_metric(metric) — one-liner chart for a metric across all companies
    4. plot_margins(num, den) — one-liner ratio/margin chart

  Standard charts (Revenue Trend, Operating Margin %, Gross Margin %) are
  auto-generated when the DB loads, guaranteeing ≥2 charts per report.

DB schema (table: financials):
    ticker       TEXT   — e.g. "NVDA"
    company      TEXT   — full company name
    period       TEXT   — ISO date of period end, e.g. "2023-01-29"
    fiscal_year  TEXT   — 4-digit year, e.g. "2023"
    form         TEXT   — "10-K" or "10-Q"
    metric       TEXT   — e.g. "revenue", "operating_income"
    value        REAL   — raw value in dollars
    value_billions REAL — value / 1e9, rounded to 3 dp

Two-stage output:
  Stage 1 — agent runs tools; results captured in _eda_observations.
  Stage 2 — fresh tiny LLM call synthesises observations → EDAFindings.
"""

from __future__ import annotations

import json
import os
import sqlite3
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.statistics import compute_statistics, group_and_filter
from tools.code_executor import execute_python, ARTIFACTS_DIR
from tools.visualizer import line_chart, bar_chart, waterfall_chart

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Data store paths ───────────────────────────────────────────────────────────

_EDA_DB_FILE: str = str(ARTIFACTS_DIR / "eda.db")
_EDA_DATA_SUMMARY: str = ""
_eda_observations: list[dict] = []


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_EDA_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def load_eda_db(all_records: list[dict]) -> None:
    """Load ALL records (10-K + 10-Q, full history) into SQLite.

    Creates the 'financials' table fresh on each pipeline run.
    This is called with the full uncompacted record set so the agent
    has access to quarterly data and full history via SQL.
    """
    conn = sqlite3.connect(_EDA_DB_FILE)
    conn.execute("DROP TABLE IF EXISTS financials")
    conn.execute("""
        CREATE TABLE financials (
            ticker        TEXT,
            company       TEXT,
            period        TEXT,
            fiscal_year   TEXT,
            form          TEXT,
            metric        TEXT,
            value         REAL,
            value_billions REAL
        )
    """)
    conn.executemany(
        "INSERT INTO financials VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                r.get("ticker", ""),
                r.get("company", ""),
                r.get("period", ""),
                r.get("fiscal_year", ""),
                r.get("form", ""),
                r.get("metric", ""),
                r.get("value"),
                r.get("value_billions"),
            )
            for r in all_records
        ],
    )
    conn.commit()
    conn.close()


def _build_series(metric: str, form: str = "10-K") -> dict[str, dict[str, float]]:
    """Build {ticker: {fiscal_year: value_billions}} for auto-charts."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT ticker, fiscal_year, value_billions FROM financials "
        "WHERE metric=? AND form=? AND value_billions IS NOT NULL "
        "ORDER BY fiscal_year",
        (metric, form),
    ).fetchall()
    conn.close()
    series: dict[str, dict[str, float]] = {}
    for row in rows:
        series.setdefault(row["ticker"], {})[row["fiscal_year"]] = row["value_billions"]
    return series


def _auto_generate_charts() -> list[dict]:
    """Pre-generate standard charts from the DB. Called by load_eda_data()."""
    auto_obs: list[dict] = []

    # Revenue trend
    rev_series = _build_series("revenue")
    if rev_series:
        path = line_chart(
            rev_series, "Revenue Trend",
            subtitle="Annual 10-K filings | USD billions",
            y_format="billions", filename="revenue_trend",
        )
        auto_obs.append({"tool": "auto_chart", "description": "Revenue Trend",
                          "value": path, "artifact_path": path})

    # Operating margin %
    rev = _build_series("revenue")
    op = _build_series("operating_income")
    if rev and op:
        margin_series: dict[str, dict[str, float]] = {}
        for ticker in set(rev) & set(op):
            for year in set(rev.get(ticker, {})) & set(op.get(ticker, {})):
                r, o = rev[ticker].get(year, 0), op[ticker].get(year, 0)
                if r:
                    margin_series.setdefault(ticker, {})[year] = round(o / r * 100, 2)
        if margin_series:
            path = line_chart(
                margin_series, "Operating Margin %",
                subtitle="Operating income / Revenue | Annual 10-K",
                y_format="pct", filename="operating_margin",
            )
            auto_obs.append({"tool": "auto_chart", "description": "Operating Margin %",
                              "value": path, "artifact_path": path})

    # Gross margin %
    gp = _build_series("gross_profit")
    if rev and gp:
        gm_series: dict[str, dict[str, float]] = {}
        for ticker in set(rev) & set(gp):
            for year in set(rev.get(ticker, {})) & set(gp.get(ticker, {})):
                r, g = rev[ticker].get(year, 0), gp[ticker].get(year, 0)
                if r:
                    gm_series.setdefault(ticker, {})[year] = round(g / r * 100, 2)
        if gm_series:
            path = line_chart(
                gm_series, "Gross Margin %",
                subtitle="Gross profit / Revenue | Annual 10-K",
                y_format="pct", filename="gross_margin",
            )
            auto_obs.append({"tool": "auto_chart", "description": "Gross Margin %",
                              "value": path, "artifact_path": path})

    return auto_obs


def load_eda_data(compact_records: list[dict]) -> str:
    """Pre-generate standard charts and return a data summary for the EDA prompt.

    NOTE: call load_eda_db(all_records) BEFORE this so the DB is populated.
    compact_records is only used to build the summary (tickers/metrics/years).
    """
    global _EDA_DATA_SUMMARY, _eda_observations

    # Read summary info from DB (more complete than compact_records)
    conn = _get_conn()
    tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM financials ORDER BY ticker").fetchall()]
    metrics = [r[0] for r in conn.execute(
        "SELECT DISTINCT metric FROM financials ORDER BY metric").fetchall()]
    years = [r[0] for r in conn.execute(
        "SELECT DISTINCT fiscal_year FROM financials WHERE form='10-K' "
        "ORDER BY fiscal_year").fetchall()]
    total = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    annual = conn.execute(
        "SELECT COUNT(*) FROM financials WHERE form='10-K'").fetchone()[0]
    conn.close()

    # Auto-generate standard charts; seed observation store
    _eda_observations = _auto_generate_charts()

    _EDA_DATA_SUMMARY = (
        f"Financial data loaded into SQLite (table: financials).\n"
        f"Total rows: {total} ({annual} annual 10-K + {total-annual} quarterly 10-Q)\n"
        f"Companies ({len(tickers)}): {', '.join(tickers)}\n"
        f"Metrics: {', '.join(metrics)}\n"
        f"Annual fiscal years available: {years[0] if years else '?'} – {years[-1] if years else '?'}\n"
        f"Standard charts pre-generated: Revenue Trend, Operating Margin %, Gross Margin %.\n"
    )
    return _EDA_DATA_SUMMARY


def clear_eda_store() -> None:
    """No-op — load_eda_data() owns the store reset."""
    pass


def get_eda_observations() -> list[dict]:
    return list(_eda_observations)


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


# ── Prompt ────────────────────────────────────────────────────────────────────

EDA_PROMPT = """\
You are the EDA (Exploratory Data Analysis) agent for a Sector Analyst system.

The financial data is in a SQLite database. Standard charts are already generated.
Your job: run deeper analysis, compute growth rates / margins / regressions,
and generate additional charts.

━━ DATA ACCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DB table: financials
Columns: ticker, company, period, fiscal_year, form, metric, value, value_billions
form values: '10-K' (annual) or '10-Q' (quarterly)
metric values: revenue, net_income, operating_income, gross_profit, rd_expense, ...

Example queries:
  SELECT ticker, fiscal_year, value_billions
  FROM financials WHERE metric='revenue' AND form='10-K'
  ORDER BY ticker, fiscal_year

  SELECT ticker, AVG(value_billions) as avg_rev
  FROM financials WHERE metric='revenue' AND form='10-K'
  GROUP BY ticker ORDER BY avg_rev DESC

━━ TOOLS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. sql_query(sql) — run any SELECT query. Returns list of row dicts.
   Use for quick lookups, aggregations, rankings.

2. run_python(code) — full Python environment: pandas, numpy, scipy, sklearn.
   conn and df are pre-loaded:
     conn = sqlite3.connect('...eda.db')
     df   = pd.read_sql("SELECT * FROM financials", conn)
   Use for: pivot tables, YoY growth, CAGR, linear regression, correlation.
   IMPORTANT: print() every computed result — outputs are captured for the report.

3. plot_metric(metric, chart_type, y_format, title)
   One-liner chart for a metric across all companies (annual 10-K).
   Examples: plot_metric("rd_expense")
             plot_metric("net_income", title="Net Income Trend")

4. plot_margins(numerator_metric, denominator_metric, title)
   Auto-computes ratio % per company per year and plots it.
   Examples: plot_margins("rd_expense", "revenue", "R&D Intensity %")
             plot_margins("net_income", "revenue", "Net Margin %")

5. create_chart(chart_type, series_data, title, subtitle, y_format, filename)
   Custom chart with manually specified data. Use for:
   - Bar chart of a single year's metric across companies
   - Waterfall chart of revenue composition
   - Custom computed series (e.g. YoY growth rates from run_python)

━━ ANALYSIS PLAN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Work through as many of these as the data supports:
□ sql_query: rank companies by average revenue
□ sql_query: find which company has the highest revenue CAGR
□ run_python: compute YoY revenue growth % per company; print a table
□ run_python: linear regression of revenue vs year for each company (slope = growth trend)
□ run_python: compute correlation matrix across metrics
□ plot_metric("rd_expense") if R&D data is available
□ plot_margins("rd_expense", "revenue", "R&D Intensity %")
□ plot_margins("net_income", "revenue", "Net Margin %")
□ Identify the single biggest outlier vs peers

Make all your tool calls. You do not need to write any final message.
"""


# ── Tool wrappers ─────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def sql_query(sql: str) -> list[dict]:
    """Run a SELECT query on the financials table. Returns list of row dicts.

    Table: financials(ticker, company, period, fiscal_year, form, metric, value, value_billions)
    Always filter form='10-K' for annual comparisons, or '10-Q' for quarterly trends.
    """
    try:
        conn = _get_conn()
        rows = conn.execute(sql).fetchall()
        conn.close()
        result = [dict(r) for r in rows]
        _eda_observations.append({
            "tool": "sql_query",
            "description": sql.strip()[:120],
            "value": str(result[:5])[:500] + (f" ... ({len(result)} rows total)" if len(result) > 5 else ""),
            "artifact_path": None,
        })
        return result
    except Exception as e:
        return [{"error": str(e), "sql": sql}]


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """Execute Python code with pandas, numpy, scipy, sklearn.

    Pre-loaded variables:
      conn  = sqlite3.connect(db_path)
      df    = pd.read_sql("SELECT * FROM financials", conn)
    df columns: ticker, company, period, fiscal_year, form, metric, value, value_billions

    Print ALL computed results to stdout — they are captured for the report.
    """
    preamble = (
        "import sqlite3, pandas as pd, numpy as np\n"
        "from scipy import stats\n"
        "try:\n"
        "    from sklearn.linear_model import LinearRegression\n"
        "except ImportError:\n"
        "    pass\n"
        f"conn = sqlite3.connect({repr(_EDA_DB_FILE)})\n"
        "df = pd.read_sql(\"SELECT * FROM financials\", conn)\n"
        "conn.close()\n"
    )
    result = execute_python(preamble + code)
    stdout = result.get("stdout", "").strip()
    _eda_observations.append({
        "tool": "run_python",
        "description": "Python analysis",
        "value": stdout[:800] if stdout else result.get("stderr", "")[:400],
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
    """Plot a financial metric for all companies (annual 10-K data only).

    metric: e.g. 'revenue', 'rd_expense', 'net_income', 'gross_profit'
    chart_type: 'line' or 'bar'
    y_format: 'billions' or 'pct' or 'raw'
    Returns: relative path to saved PNG.
    """
    series = _build_series(metric)
    if not series:
        return f"No data for metric='{metric}'"
    chart_title = title or metric.replace("_", " ").title()
    fn = bar_chart if chart_type == "bar" else line_chart
    path = fn(series, chart_title, subtitle="Annual 10-K filings",
               y_format=y_format, filename=metric)
    _eda_observations.append({
        "tool": "plot_metric", "description": chart_title,
        "value": path, "artifact_path": path,
    })
    return path


@function_tool(strict_mode=False)
def plot_margins(
    numerator_metric: str,
    denominator_metric: str,
    title: str | None = None,
) -> str:
    """Compute and plot a margin/ratio (numerator/denominator * 100) per company per year.

    Examples:
      plot_margins('rd_expense', 'revenue', 'R&D Intensity %')
      plot_margins('net_income', 'revenue', 'Net Margin %')
    Returns: relative path to saved PNG.
    """
    num = _build_series(numerator_metric)
    den = _build_series(denominator_metric)
    if not num or not den:
        return f"No data for '{numerator_metric}' or '{denominator_metric}'"
    series: dict[str, dict[str, float]] = {}
    for ticker in set(num) & set(den):
        for year in set(num.get(ticker, {})) & set(den.get(ticker, {})):
            d = den[ticker].get(year, 0)
            if d:
                series.setdefault(ticker, {})[year] = round(num[ticker][year] / d * 100, 2)
    if not series:
        return "No overlapping data to compute margin"
    chart_title = title or f"{numerator_metric} / {denominator_metric} %"
    path = line_chart(series, chart_title, subtitle="Annual 10-K filings",
                      y_format="pct",
                      filename=f"{numerator_metric}_over_{denominator_metric}")
    _eda_observations.append({
        "tool": "plot_margins", "description": chart_title,
        "value": path, "artifact_path": path,
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
    """Custom chart with manually specified series_data.
    Use for waterfall, bar, or custom computed series (e.g. YoY growth rates).
    series_data: {series_name: {x_label: y_value}}
    """
    if chart_type == "line":
        path = line_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "bar":
        path = bar_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    elif chart_type == "waterfall":
        path = waterfall_chart(series_data, title, subtitle=subtitle, y_format=y_format, filename=filename)
    else:
        return f"Unknown chart_type '{chart_type}'"
    _eda_observations.append({
        "tool": "create_chart", "description": title,
        "value": path, "artifact_path": path,
    })
    return path


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_eda_agent() -> Agent:
    return Agent(
        name="EDA",
        model=_make_model(),
        instructions=EDA_PROMPT,
        tools=[sql_query, run_python, plot_metric, plot_margins, create_chart],
    )
