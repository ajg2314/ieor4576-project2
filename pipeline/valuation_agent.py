"""Valuation Agent — builds a peer comp table and interprets relative valuations.

Step 1.5 (runs in parallel with Researcher and Sentiment Agent).

Uses live market data (yfinance via tools/market_data.py) to:
  - Fetch current P/E, EV/EBITDA, EV/Revenue, P/B, YTD return, analyst consensus
  - Build a sector comp table across all peer tickers
  - Compute sector median multiples
  - Identify which companies look expensive vs. cheap relative to peers and why

Output: ValuationContext schema — fed directly into the Hypothesis agent's Valuation section.
"""

from __future__ import annotations

import os
import statistics
from datetime import date

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.market_data import (
    get_sector_market_data as _get_sector_market_data,
    get_ytd_return as _get_ytd_return,
)

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Side-channel observations ─────────────────────────────────────────────────
_valuation_observations: list[dict] = []


def get_valuation_observations() -> list[dict]:
    return list(_valuation_observations)


def clear_valuation_observations() -> None:
    global _valuation_observations
    _valuation_observations = []


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def fetch_peer_valuations(tickers: list[str]) -> list[dict]:
    """Fetch current valuation multiples for a list of tickers.

    Returns a list of dicts containing: ticker, company_name, current_price,
    market_cap_b, pe_trailing, pe_forward, ev_ebitda, ev_revenue, price_to_book,
    analyst_recommendation, price_target. None values mean data was unavailable.
    """
    results = _get_sector_market_data(tickers)
    _valuation_observations.append({
        "tool": "fetch_peer_valuations",
        "description": f"Valuation multiples for {len(tickers)} tickers",
        "value": f"Fetched data for: {', '.join(r.get('ticker','?') for r in results)}",
        "artifact_path": None,
    })
    return results


@function_tool(strict_mode=False)
def fetch_ytd_returns(tickers: list[str]) -> dict:
    """Fetch year-to-date price return (%) for each ticker.

    Returns a dict mapping ticker -> ytd_return_pct (float or null if unavailable).
    YTD is computed from the first trading day of the current calendar year.
    """
    returns = {}
    for ticker in tickers:
        returns[ticker.upper()] = _get_ytd_return(ticker)
    _valuation_observations.append({
        "tool": "fetch_ytd_returns",
        "description": "YTD price returns",
        "value": str({k: (f"{v:+.1f}%" if v is not None else "N/A") for k, v in returns.items()}),
        "artifact_path": None,
    })
    return returns


# ── Agent ─────────────────────────────────────────────────────────────────────

VALUATION_PROMPT = f"""\
You are the Valuation Agent in a multi-agent Sector Analyst system.
Today's date: {date.today().isoformat()}

You receive a list of tickers for a sector and your job is to:
1. Fetch live valuation data for all tickers
2. Build a comp table
3. Compute sector median multiples
4. Interpret relative valuations

STEP 1 — Call fetch_peer_valuations(tickers) to get all valuation multiples.
STEP 2 — Call fetch_ytd_returns(tickers) to get year-to-date price performance.
STEP 3 — Output the ValuationContext JSON.

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After ALL tool calls complete, output ONLY this JSON (no markdown fences):

{{
  "sector": "<sector name>",
  "as_of_date": "{date.today().isoformat()}",
  "metrics": [
    {{
      "ticker": "NVDA",
      "company_name": "NVIDIA Corporation",
      "current_price": 875.40,
      "market_cap_b": 2157.3,
      "pe_trailing": 55.2,
      "pe_forward": 32.1,
      "ev_ebitda": 48.7,
      "ev_revenue": 28.4,
      "price_to_book": 42.1,
      "ytd_return_pct": 82.5,
      "analyst_recommendation": "buy",
      "price_target": 1050.0
    }}
  ],
  "sector_median_pe": <median of all non-null pe_trailing values, or null>,
  "sector_median_ev_ebitda": <median of all non-null ev_ebitda values, or null>,
  "summary": "<2-3 paragraphs: who looks expensive vs cheap relative to peers, what the valuation spread implies about growth expectations, which companies have the most/least compelling risk-reward based on multiples vs growth rates>"
}}

RULES:
- Use ONLY data from the tool results. Do not invent numbers.
- null means the data was genuinely unavailable — use it, don't fabricate.
- Compute sector_median_pe yourself from the pe_trailing values you received.
- In the summary: connect valuation to growth. A high P/E is not expensive if the
  growth rate justifies it (PEG ratio logic). A low P/E is not cheap if growth is negative.
- Identify 1-2 companies that look most attractively valued and explain why.
- Identify 1-2 companies that look most expensive relative to peers and explain why.
"""


def build_valuation_agent() -> Agent:
    """Build the Valuation agent. Call build_valuation_agent() fresh each pipeline run."""
    return Agent(
        name="Valuation",
        model=LitellmModel(model=LITELLM_MODEL_ID),
        instructions=VALUATION_PROMPT,
        tools=[fetch_peer_valuations, fetch_ytd_returns],
    )
