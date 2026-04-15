"""Market data tools — yfinance wrapper for Valuation and Peer Discovery pipeline steps.

Uses an in-memory cache keyed by ticker to avoid redundant network calls within
a single pipeline run. Cache is intentionally NOT persisted across runs (stale data risk).

Design notes:
- yfinance is imported lazily inside each function (never at module top) so the
  rest of the system works even if yfinance is not installed.
- All public functions are plain Python (not @function_tool) — they are called
  by @function_tool wrappers in valuation_agent.py or directly from peer_discovery.py.
- Follows the same side-channel observation pattern as tools/sec_edgar.py.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")

logger = logging.getLogger(__name__)

# ── In-memory per-run cache ───────────────────────────────────────────────────
_info_cache: dict[str, dict] = {}
_history_cache: dict[str, Any] = {}

# ── Side-channel observations (follows sec_edgar pattern) ─────────────────────
_observations: list[dict] = []


def get_market_data_observations() -> list[dict]:
    """Return all observations captured during this run."""
    return list(_observations)


def clear_market_data_observations() -> None:
    """Reset observations and cache between pipeline runs."""
    global _observations, _info_cache, _history_cache
    _observations = []
    _info_cache = {}
    _history_cache = {}


# ── Core data fetching ────────────────────────────────────────────────────────

def _fetch_ticker_info(ticker: str) -> dict:
    """Fetch yfinance .info dict with in-memory cache. Returns {} on any failure.

    yfinance is imported lazily here so the module can be imported even when
    yfinance is not installed (e.g. during test collection).
    """
    ticker = ticker.upper()
    if ticker in _info_cache:
        return _info_cache[ticker]
    try:
        import yfinance as yf  # noqa: PLC0415 — lazy import intentional
        t = yf.Ticker(ticker)
        info = t.info or {}
        _info_cache[ticker] = info
        return info
    except Exception as exc:
        logger.warning("yfinance .info failed for %s: %s", ticker, exc)
        _info_cache[ticker] = {}
        return {}


def get_stock_info(ticker: str) -> dict[str, Any]:
    """Return price, market cap, valuation multiples, sector, and analyst data for one ticker.

    All fields use .get() with None defaults — many fields are absent for foreign ADRs.
    Currency note: for non-USD tickers (TSM, ASML, SAP), market_cap is in local currency;
    a warning is logged and the caller should treat these values as approximate.
    """
    ticker = ticker.upper()
    info = _fetch_ticker_info(ticker)

    currency = info.get("currency", "USD")
    if currency and currency != "USD":
        logger.warning("Ticker %s reports in %s — market cap values are in local currency", ticker, currency)

    market_cap = info.get("marketCap")
    market_cap_b = round(market_cap / 1e9, 3) if market_cap else None

    result: dict[str, Any] = {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": market_cap,
        "market_cap_b": market_cap_b,
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_revenue": info.get("enterpriseToRevenue"),
        "price_to_book": info.get("priceToBook"),
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "analyst_recommendation": info.get("recommendationKey"),
        "price_target": info.get("targetMeanPrice"),
        "currency": currency,
        "valid": bool(market_cap),  # True only if we got actual market data
    }

    _observations.append({
        "tool": "get_stock_info",
        "ticker": ticker,
        "description": f"{ticker} market data",
        "value": f"market_cap=${market_cap_b}B, P/E={result['pe_trailing']}, recommendation={result['analyst_recommendation']}",
        "artifact_path": None,
    })
    return result


def get_sector_market_data(tickers: list[str]) -> list[dict[str, Any]]:
    """Batch fetch stock info for multiple tickers. Returns list in same order as input."""
    results = []
    for ticker in tickers:
        try:
            results.append(get_stock_info(ticker))
        except Exception as exc:
            logger.warning("get_sector_market_data failed for %s: %s", ticker, exc)
            results.append({"ticker": ticker.upper(), "valid": False, "error": str(exc)})
    return results


def get_ytd_return(ticker: str) -> float | None:
    """Compute YTD price return as a percentage. Returns None on failure.

    YTD = (current price - price on first trading day of year) / price on first trading day * 100
    """
    ticker = ticker.upper()
    try:
        import yfinance as yf  # noqa: PLC0415
        t = yf.Ticker(ticker)
        today = date.today()
        start_of_year = f"{today.year}-01-01"
        hist = t.history(start=start_of_year, auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return None
        first_close = float(hist["Close"].iloc[0])
        last_close = float(hist["Close"].iloc[-1])
        if first_close == 0:
            return None
        ytd_pct = round((last_close - first_close) / first_close * 100, 2)
        _observations.append({
            "tool": "get_ytd_return",
            "ticker": ticker,
            "description": f"{ticker} YTD return",
            "value": f"{ytd_pct:+.1f}%",
            "artifact_path": None,
        })
        return ytd_pct
    except Exception as exc:
        logger.warning("YTD return failed for %s: %s", ticker, exc)
        return None


def get_company_financials_yf(ticker: str, concepts: list[str] | None = None) -> dict:
    """Fetch annual income statement data via yfinance for non-EDGAR companies.

    Returns flat_records in the same schema as SEC EDGAR get_company_financials(),
    so the EDA agent can load them into SQLite without any changes.
    """
    import yfinance as yf  # noqa: PLC0415
    concepts = concepts or ["revenue", "net_income", "operating_income", "gross_profit", "rd_expense"]
    col_map = {
        "Total Revenue": "revenue",
        "Gross Profit": "gross_profit",
        "Operating Income": "operating_income",
        "Net Income": "net_income",
        "Research And Development": "rd_expense",
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        company_name = info.get("longName") or info.get("shortName") or ticker
        fin = t.income_stmt
        if fin is None or fin.empty:
            fin = t.financials
        if fin is None or fin.empty:
            return {"ticker": ticker, "company_name": ticker, "flat_records": [],
                    "source": "yfinance", "error": "no data"}
        flat_records = []
        for col in fin.columns:
            try:
                period_str = col.strftime("%Y-%m-%d")
                fiscal_year = str(col.year)
            except Exception:
                continue
            for yf_col, concept in col_map.items():
                if concept not in concepts or yf_col not in fin.index:
                    continue
                raw_val = fin.loc[yf_col, col]
                try:
                    value = float(raw_val)
                    if value != value:  # NaN
                        continue
                except (TypeError, ValueError):
                    continue
                flat_records.append({
                    "ticker": ticker, "company": company_name,
                    "period": period_str, "fiscal_year": fiscal_year,
                    "form": "yfinance-annual", "metric": concept,
                    "value": value, "value_billions": round(value / 1e9, 3),
                })
        return {"ticker": ticker, "company_name": company_name,
                "flat_records": flat_records, "source": "yfinance"}
    except Exception as e:
        return {"ticker": ticker, "company_name": ticker, "flat_records": [],
                "source": "yfinance", "error": str(e)}


def get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Return price history as a serialisable dict.

    Args:
        ticker: Stock ticker symbol.
        period: yfinance period string — '1mo', '3mo', '6mo', '1y', '2y', '5y'.

    Returns:
        {"ticker": str, "period": str, "data": {date_str: close_price}}
    """
    ticker = ticker.upper()
    cache_key = f"{ticker}_{period}"
    if cache_key in _history_cache:
        return _history_cache[cache_key]
    try:
        import yfinance as yf  # noqa: PLC0415
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            result: dict[str, Any] = {"ticker": ticker, "period": period, "data": {}}
        else:
            result = {
                "ticker": ticker,
                "period": period,
                "data": {
                    str(idx.date()): round(float(row["Close"]), 2)
                    for idx, row in hist.iterrows()
                },
            }
        _history_cache[cache_key] = result
        return result
    except Exception as exc:
        logger.warning("Price history failed for %s (%s): %s", ticker, period, exc)
        result = {"ticker": ticker, "period": period, "data": {}, "error": str(exc)}
        _history_cache[cache_key] = result
        return result
