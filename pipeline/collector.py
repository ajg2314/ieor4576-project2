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
)
from models.schemas import DataBundle

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


COLLECTOR_PROMPT = """\
You are the Data Collector agent for a Sector Analyst system.

Your job is to retrieve real SEC filing data for the companies or sector
the user is asking about. You have access to two distinct data sources:

RETRIEVAL METHOD 1 — EDGAR XBRL Structured Financial Data:
- Use `lookup_ticker` to resolve a company name or ticker to its SEC CIK number.
- Use `fetch_company_financials` to retrieve structured financial data for one company
  (revenue, net income, operating income, gross profit, R&D, etc.) from SEC filings.
- Use `fetch_sector_financials` to retrieve data for multiple companies at once.
- Available financial concepts: revenue, net_income, operating_income, gross_profit,
  operating_expenses, eps, total_assets, total_debt, cash, rd_expense

RETRIEVAL METHOD 2 — Filing Text Scraping (MD&A):
- Use `fetch_filing_text` to scrape the Management Discussion & Analysis (MD&A)
  section from a company's most recent 10-K or 10-Q filing.
- This provides qualitative context: guidance, risk factors, segment commentary.
- Use this for at least one company to complement the quantitative data.

PROCESS:
1. Parse the user's question to identify the sector or companies of interest.
2. If the user mentions a sector (e.g. "semiconductors"), identify 3–5 major companies.
   Use your knowledge of which companies belong to the sector, then verify tickers.
3. Call `fetch_sector_financials` for the full company basket.
4. Call `fetch_filing_text` for at least one key company to get qualitative data.
5. Return a structured DataBundle summarizing what you retrieved.

The DataBundle's `records` field should be a flat list of financial data points
suitable for the EDA agent to analyze.

OUTPUT FORMAT: Your final response must be a single JSON object with these exact keys:
{
  "source": "<name/URL of primary data source>",
  "retrieval_method": "<api|sql|web|rag>",
  "records": [{"ticker": "...", "period": "...", "metric": "...", "value": ...}, ...],
  "metadata": {"time_range": "...", "companies": [...], ...},
  "summary": "<short description of what was retrieved and why>"
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


def build_collector_agent() -> Agent:
    return Agent(
        name="Collector",
        model=_make_model(),
        instructions=COLLECTOR_PROMPT,
        tools=[lookup_ticker, fetch_company_financials, fetch_sector_financials, fetch_filing_text],
    )
