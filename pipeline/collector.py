"""Collector Agent — retrieves real-world data from SEC EDGAR.

Step 1: Collect. Two retrieval methods:
  1. EDGAR XBRL API   — structured financial data (revenue, income, margins) per company.
                        The raw company facts JSON is hundreds of KB; the agent queries
                        specific concepts and periods rather than loading everything.
  2. Filing text scraping — fetches and extracts the MD&A section of actual 10-K/10-Q
                            documents for qualitative signals and management commentary.
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.sec_edgar import (
    resolve_ticker,
    search_companies_by_name,
    get_company_financials,
    get_sector_financials,
    get_recent_filing_text,
    append_to_record_store,
)
from tools.market_data import get_company_financials_yf as _get_yf_financials
from models.schemas import DataBundle

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


COLLECTOR_PROMPT = """\
You are the Data Collector agent for a Sector Analyst system.

You will receive a research brief that already identifies the sector and the exact
list of tickers to fetch. Your job is purely data retrieval — do NOT decide which
companies to include; that was decided by the Planner. Follow the brief exactly.

RETRIEVAL METHOD 1 — EDGAR XBRL Structured Financial Data:
- Use `fetch_sector_financials` to retrieve data for ALL tickers listed in the brief.
  Pass every ticker from the list — do not drop any.
- Available financial concepts: revenue, net_income, operating_income, gross_profit,
  operating_expenses, eps, total_assets, total_debt, cash, rd_expense
- Use the focus_metrics listed in the brief. If none are listed, default to:
  revenue, net_income, operating_income, gross_profit, rd_expense

RETRIEVAL METHOD 2 — Filing Text Scraping (MD&A):
- Use `fetch_filing_text` for the FIRST company in the ticker list to get qualitative
  context: guidance, risk factors, segment commentary.
- Fetch BOTH the most recent 10-K (annual) AND the most recent 10-Q (quarterly) for
  the first ticker. The 10-Q has more recent management commentary and forward guidance.

PROCESS:
1. Read the tickers and focus_metrics from the research brief.
2. Call `fetch_sector_financials` with all tickers and focus_metrics.
3. For any ticker where `fetch_sector_financials` returns 0 records (common for European-listed
   companies such as DNNGY, VWDRY, SMEGF, or any OTC pink-sheet ADR), immediately call
   `fetch_company_financials_yf` for that ticker as a fallback.
4. Call `fetch_filing_text` for the first ticker with form_type="10-K" (annual report).
5. Call `fetch_filing_text` for the first ticker with form_type="10-Q" (most recent quarter).
6. Output the DataBundle metadata — do NOT include any records in the JSON.

FALLBACK FOR NON-US / NON-EDGAR COMPANIES:
If `fetch_sector_financials` or `fetch_company_financials` returns 0 records for a ticker
(common for European-listed companies like DNNGY, VWDRY, SMEGF, or any OTC pink-sheet ADR),
immediately call `fetch_company_financials_yf` for that ticker. It covers international
stocks with the same data schema. Always try EDGAR first; use yfinance only as fallback.

IMPORTANT — records field:
- Leave "records" as an empty array [].
- Records are automatically captured from your tool calls via a side-channel.
- Only populate metadata and summary.

OUTPUT FORMAT: Your final response must be a single JSON object. No other text.
{
  "source": "SEC EDGAR XBRL API",
  "retrieval_method": "api",
  "records": [],
  "metadata": {"companies": [...tickers fetched...], "concepts": [...metrics...], "mda_summary": "...(10-K + 10-Q combined)"},
  "summary": "<what was retrieved: company names, metrics, date range>"
}
IMPORTANT: After all tool calls are complete, you MUST send one final text message
containing ONLY the JSON object above. Do not stop after the last tool call.
"""


@function_tool(strict_mode=False)
def lookup_ticker(ticker_or_name: str) -> dict:
    """Look up a stock ticker to get its SEC CIK number and full company name."""
    try:
        return resolve_ticker(ticker_or_name)
    except ValueError:
        return search_companies_by_name(ticker_or_name, max_results=5)


@function_tool(strict_mode=False)
def fetch_company_financials(ticker: str, concepts: list[str] | None = None) -> dict:
    """
    Retrieve structured XBRL financial data for one company from SEC EDGAR.
    concepts can include: revenue, net_income, operating_income, gross_profit,
    operating_expenses, eps, total_assets, total_debt, cash, rd_expense
    """
    return get_company_financials(ticker, concepts)


@function_tool(strict_mode=False)
def fetch_sector_financials(tickers: list[str], concepts: list[str] | None = None) -> dict:
    """
    Retrieve financial data for multiple companies (a sector basket) from SEC EDGAR.
    """
    return get_sector_financials(tickers, concepts)


@function_tool(strict_mode=False)
def fetch_filing_text(ticker: str, form_type: str = "10-K") -> dict:
    """
    Scrape and extract the MD&A section from a company's most recent SEC filing.
    form_type: '10-K' (annual) or '10-Q' (quarterly)
    """
    return get_recent_filing_text(ticker, form_type)


@function_tool(strict_mode=False)
def fetch_company_financials_yf(ticker: str) -> dict:
    """Fetch annual income statement data via yfinance — use as fallback for non-EDGAR companies.

    Use this when fetch_company_financials() or fetch_sector_financials() returns 0 records
    (e.g. European-listed companies like DNNGY, VWDRY, SMEGF). Returns same flat_records
    schema as EDGAR so the EDA agent can load them into SQLite without any changes.
    """
    result = _get_yf_financials(ticker)
    records = result.get("flat_records", [])
    append_to_record_store(records)  # same side-channel as SEC EDGAR tools
    return {
        "ticker": ticker,
        "company_name": result.get("company_name", ticker),
        "records_fetched": len(records),
        "source": "yfinance",
        "error": result.get("error"),
    }


def build_collector_agent() -> Agent:
    return Agent(
        name="Collector",
        model=_make_model(),
        instructions=COLLECTOR_PROMPT,
        tools=[
            lookup_ticker, fetch_company_financials, fetch_sector_financials,
            fetch_filing_text, fetch_company_financials_yf,
        ],
    )
