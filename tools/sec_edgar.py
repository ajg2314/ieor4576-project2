"""Tools: SEC EDGAR data retrieval for the Collector agent.

Two retrieval methods:
1. EDGAR XBRL API  — structured financial data (revenue, income, margins) per company.
   Returns large JSON datasets that are filtered/queried rather than dumped into context.
2. Filing text scraper — fetches the MD&A section of recent 10-K/10-Q documents
   for qualitative signals (risk factors, management commentary, guidance).

SEC EDGAR is a public API. No API key required.
Rate limit: 10 req/s. User-Agent header is required by SEC policy.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

# SEC requires a descriptive User-Agent with contact info
HEADERS = {
    "User-Agent": "IEOR4576 Project research@columbia.edu",
    "Accept-Encoding": "gzip, deflate",
}

# ---------------------------------------------------------------------------
# Side-channel record store
# The orchestrator reads from here after the collector agent finishes, so the
# LLM never needs to echo the full records array in its text output.
# ---------------------------------------------------------------------------

_record_store: list[dict] = []


def clear_record_store() -> None:
    """Clear the store before each new pipeline run."""
    global _record_store
    _record_store = []


def get_stored_records() -> list[dict]:
    """Return deduplicated records accumulated by tool calls.

    Deduplicates on (ticker, period, metric) — last write wins, so if the
    collector called fetch_company_financials and fetch_sector_financials for
    the same ticker, only one record per data point is returned.
    """
    seen: dict[tuple, dict] = {}
    for r in _record_store:
        key = (r.get("ticker"), r.get("period"), r.get("metric"))
        seen[key] = r  # last write wins
    return list(seen.values())

# Small rate-limiting delay between requests
_last_request_time = 0.0
MIN_INTERVAL = 0.12  # ~8 req/s, safely under 10/s limit


def _get(url: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    """Throttled GET with required SEC headers."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    resp = httpx.get(url, params=params, headers=HEADERS, timeout=timeout, follow_redirects=True)
    _last_request_time = time.monotonic()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Company lookup
# ---------------------------------------------------------------------------

_ticker_map: dict[str, dict] | None = None


def _load_ticker_map() -> dict[str, dict]:
    """Load SEC's full ticker→CIK mapping (cached in memory)."""
    global _ticker_map
    if _ticker_map is None:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        # Keys are stringified indices; values have cik_str, ticker, title
        _ticker_map = {v["ticker"].upper(): v for v in data.values()}
    return _ticker_map


def resolve_ticker(ticker: str) -> dict[str, Any]:
    """
    Resolve a stock ticker to its SEC CIK number and company name.

    Args:
        ticker: Stock ticker symbol (e.g. 'NVDA', 'AAPL')

    Returns:
        dict with 'cik', 'cik_padded' (10-digit), 'ticker', 'title'
    """
    mapping = _load_ticker_map()
    entry = mapping.get(ticker.upper())
    if not entry:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR. Try the exact ticker symbol.")
    cik = int(entry["cik_str"])
    return {
        "cik": cik,
        "cik_padded": str(cik).zfill(10),
        "ticker": entry["ticker"],
        "title": entry["title"],
    }


def search_companies_by_name(name: str, max_results: int = 10) -> list[dict[str, Any]]:
    """
    Search for companies by name on EDGAR.

    Args:
        name: Company name or partial name
        max_results: Max results to return

    Returns:
        List of dicts with 'cik', 'name', 'ticker' fields
    """
    resp = _get(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": f'"{name}"', "forms": "10-K", "hits.hits.total.value": 1},
    )
    # Also try the company search endpoint
    resp2 = _get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={
            "company": name,
            "CIK": "",
            "type": "10-K",
            "dateb": "",
            "owner": "include",
            "count": max_results,
            "search_text": "",
            "action": "getcompany",
            "output": "atom",
        },
    )
    # Parse atom XML for company entries
    results = []
    entries = re.findall(r"<entity-name>(.*?)</entity-name>.*?<cik>(.*?)</cik>", resp2.text, re.DOTALL)
    for title, cik in entries[:max_results]:
        results.append({"cik": int(cik.strip()), "name": title.strip(), "ticker": None})
    return results


# ---------------------------------------------------------------------------
# Retrieval method 1: XBRL structured financial data
# ---------------------------------------------------------------------------

# Common US-GAAP concepts for financial analysis
FINANCIAL_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_expenses": ["OperatingExpenses"],
    "eps": ["EarningsPerShareBasic", "EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
}


def get_company_financials(ticker: str, concepts: list[str] | None = None) -> dict[str, Any]:
    """
    Retrieve structured financial data for a company from SEC EDGAR XBRL API.

    This fetches the company's full financial fact history — revenue, net income,
    operating income, etc. — in structured form. The raw data is large (100s of KB)
    and is filtered to annual/quarterly figures before returning.

    Args:
        ticker: Stock ticker (e.g. 'NVDA')
        concepts: List of financial concept keys from FINANCIAL_CONCEPTS.
                  Defaults to ['revenue', 'net_income', 'operating_income', 'gross_profit']

    Returns:
        dict with company info and a 'financials' dict mapping concept → list of periods+values
    """
    if concepts is None:
        concepts = ["revenue", "net_income", "operating_income", "gross_profit"]

    company = resolve_ticker(ticker)
    cik_padded = company["cik_padded"]

    # Fetch full company facts (large JSON — filtered below, not dumped into context)
    facts_data = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json").json()
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

    financials: dict[str, list[dict]] = {}
    for concept_key in concepts:
        candidate_tags = FINANCIAL_CONCEPTS.get(concept_key, [concept_key])
        # Merge across ALL matching tags — companies change tag names over time
        # (e.g. Apple switched from Revenues → RevenueFromContractWithCustomer in FY2019)
        merged: dict[str, dict] = {}  # period → record, later tags win on overlap
        for tag in candidate_tags:
            if tag not in us_gaap:
                continue
            units = us_gaap[tag].get("units", {})
            series = units.get("USD", units.get("shares", []))
            annual = [
                {
                    "period": r["end"],
                    "value": r["val"],
                    "form": r.get("form", ""),
                    "filed": r.get("filed", ""),
                    "tag": tag,
                }
                for r in series
                if r.get("form") in ("10-K", "10-Q") and r.get("val") is not None
            ]
            # Within each tag, keep the most-recently-filed record per period.
            # Sorting ascending by filed means the last write wins per period,
            # so amended/restated filings replace the original.
            for rec in sorted(annual, key=lambda x: x["filed"]):
                merged[rec["period"]] = rec  # always overwrite → latest filing wins
        if merged:
            financials[concept_key] = sorted(merged.values(), key=lambda x: x["period"])[-40:]

    # Build flat records — one row per (ticker, period, metric)
    # This is the format the EDA agent consumes directly
    flat_records = []
    for concept_key, series in financials.items():
        for rec in series:
            flat_records.append({
                "ticker": company["ticker"],
                "company": company["title"],
                "period": rec["period"],
                "fiscal_year": rec["period"][:4],
                "form": rec["form"],
                "metric": concept_key,
                "value": rec["value"],
                "value_billions": round(rec["value"] / 1e9, 3) if isinstance(rec["value"], (int, float)) else rec["value"],
            })

    # Push records into the side-channel store so the orchestrator can
    # retrieve them without the LLM needing to echo them in its text output.
    _record_store.extend(flat_records)

    return {
        "ticker": company["ticker"],
        "company_name": company["title"],
        "cik": company["cik"],
        "financials": financials,
        "flat_records": flat_records,
        "available_concepts": list(us_gaap.keys())[:50],
    }


def get_sector_financials(tickers: list[str], concepts: list[str] | None = None) -> dict[str, Any]:
    """
    Retrieve financial data for multiple companies (a sector basket).

    Args:
        tickers: List of ticker symbols
        concepts: Financial concepts to retrieve per company

    Returns:
        dict mapping ticker → financial data, plus sector-level summary
    """
    results: dict[str, Any] = {}
    errors: list[str] = []
    for ticker in tickers:
        try:
            results[ticker] = get_company_financials(ticker, concepts)
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    # Aggregate all flat records across companies for easy EDA consumption
    all_flat = []
    for company_data in results.values():
        all_flat.extend(company_data.get("flat_records", []))

    return {
        "companies": results,
        "flat_records": all_flat,
        "errors": errors,
        "ticker_count": len(results),
    }


# ---------------------------------------------------------------------------
# Retrieval method 2: Filing text scraping (MD&A extraction)
# ---------------------------------------------------------------------------

def get_recent_filing_text(ticker: str, form_type: str = "10-K", section: str = "mda") -> dict[str, Any]:
    """
    Fetch and extract text from a company's most recent SEC filing.

    This scrapes the actual filing document — the Management Discussion & Analysis
    (MD&A) section contains forward-looking statements, risk factors, and segment
    commentary that complement the structured XBRL data.

    Args:
        ticker: Stock ticker
        form_type: Filing type ('10-K' for annual, '10-Q' for quarterly)
        section: Which section to extract. 'mda' = Management Discussion & Analysis.

    Returns:
        dict with 'ticker', 'form_type', 'filed_date', 'extracted_text' (truncated to ~6000 chars)
    """
    company = resolve_ticker(ticker)
    cik_padded = company["cik_padded"]

    # Get list of recent filings
    submissions = _get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json").json()
    filings = submissions.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    filed_dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])

    # Find most recent filing of the requested type
    target_idx = None
    for i, form in enumerate(forms):
        if form == form_type:
            target_idx = i
            break

    if target_idx is None:
        return {"ticker": ticker, "error": f"No {form_type} found for {ticker}"}

    accession = accession_numbers[target_idx].replace("-", "")
    filed_date = filed_dates[target_idx]
    primary_doc = primary_docs[target_idx]

    # Fetch the filing index to find the main document
    index_url = f"https://www.sec.gov/Archives/edgar/data/{company['cik']}/{accession}/{primary_doc}"
    doc_resp = _get(index_url)

    # Strip HTML tags
    raw_text = re.sub(r"<[^>]+>", " ", doc_resp.text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    # Extract MD&A section (heuristic: find "Management" ... next major section)
    extracted = _extract_mda_section(raw_text) if section == "mda" else raw_text[:8000]

    return {
        "ticker": ticker,
        "company_name": company["title"],
        "form_type": form_type,
        "filed_date": filed_date,
        "filing_url": index_url,
        "extracted_text": extracted[:6000],  # Truncate to avoid flooding context
        "full_text_length": len(raw_text),
    }


def _extract_mda_section(text: str) -> str:
    """Heuristically extract the MD&A section from filing text."""
    # Common MD&A header patterns
    mda_pattern = re.compile(
        r"(management.{0,30}discussion.{0,30}analysis|item\s+7\b)",
        re.IGNORECASE,
    )
    next_section_pattern = re.compile(
        r"(item\s+[89]|quantitative.{0,30}qualitative|financial\s+statements)",
        re.IGNORECASE,
    )

    match = mda_pattern.search(text)
    if not match:
        return text[:5000]

    start = match.start()
    end_match = next_section_pattern.search(text, start + 200)
    end = end_match.start() if end_match else start + 8000

    return text[start:end][:6000]
