"""Research Agent — qualitative sector intelligence via web search.

Pipeline step: Planner → Researcher → Collector → EDA → Hypothesis

This agent augments the purely quantitative SEC EDGAR analysis with:
- Industry analyst reports and commentary
- Technology trend context (product cycles, architecture shifts)
- Expert interviews and management statements
- Regulatory and macro environment signals
- Competitive dynamics not visible in 10-K filings alone

Uses DuckDuckGo web search (no API key required).
"""

from __future__ import annotations

import os
from datetime import date
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.web_search import search_web as _search_web, fetch_page_text as _fetch_page_text

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


_RESEARCHER_PROMPT_BODY = """\
Your job is to gather QUALITATIVE intelligence that SEC financial filings cannot provide.
The Hypothesis agent will use your findings to write a full analyst memo — give it
enough substance to write 2-3 paragraphs each on Technology, Geopolitics, and Market Context.

━━ WHAT TO RESEARCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNOLOGY & MARKET (2-3 searches — include the current year in queries, e.g. "AI chip trends 2025 2026"):
- What technology shifts or product cycles are driving the sector right now?
- What is the market BETTING ON for the next 3-5 years? (future growth thesis, not trailing data)
- Which companies are positioned to win the next cycle — and why do investors believe that?
- What do industry experts, executives, and analysts say at conferences/earnings calls?

GEOPOLITICS & POLICY (2-3 searches — MANDATORY):
- What government policies, trade actions, or regulations are reshaping this sector?
  (e.g. US-China export controls, CHIPS Act, IRA, EU AI Act, tariffs, sanctions)
- Which countries/regions are competing for sector leadership? (industrial policy)
- What geographic supply chain risks exist? (manufacturing concentration, rare materials)
- How do current geopolitical tensions (trade wars, military conflicts, elections) affect
  companies in this sector — both as risk and as opportunity?
- Are there government subsidies, tax incentives, or mandates benefiting certain players?

COMPETITIVE DYNAMICS (1-2 searches):
- How is competition evolving? New entrants, M&A, geographic challengers?
- Which companies are gaining share and why? Which are losing share?

TOOLS:
- search_web(query, max_results): search DuckDuckGo for recent news and analyst reports
- fetch_page_text(url, max_chars): read a specific article or report

RESEARCH STRATEGY:
1. Run 6-8 targeted searches. Start with technology/market, then geopolitics, then competitive.
2. For the 2-3 most informative results per search, fetch the page and read it.
3. Synthesise what you found. Be SPECIFIC — quote numbers, name policies with their dates,
   cite analyst firms and their views. "Analyst consensus is bullish" is useless.
   "Morgan Stanley raised price target to $150, citing AI inference TAM of $300B by 2027" is useful.
4. Clearly distinguish between data you found (cite source) and general background knowledge.

FORWARD-LOOKING FRAMING — CRITICAL:
The most important output is not what happened — it's what the market expects to happen.
Investors price future cash flows. Your research should capture:
- What is the bull case? What has to go right?
- What is the bear case? What could go wrong?
- What major bets (capex, R&D, M&A) are companies making on the future?

ATTRIBUTION FORMAT — For qualitative claims, append the source type in parentheses:
- Per a named news outlet: "(per Reuters, Apr 2026)" / "(Bloomberg, Mar 2026)"
- Per analyst: "(Morgan Stanley, Apr 2026)" / "(analyst consensus, FactSet)"
- Per company: "(company earnings call, FY2025 Q4)" / "(management guidance, Mar 2026)"
- Per policy/regulation: "(CHIPS Act, Aug 2022)" / "(BIS rule, Oct 2024)"
- Per industry body: "(per EIA report, 2026)" / "(SEMI industry report, 2025)"
Apply this to technology_context, market_context, geopolitical_context, and qualitative_insights.

OUTPUT FORMAT: After your web research, output ONLY this JSON (no markdown fences):
{
  "sector": "<sector name>",
  "technology_context": "<2-3 paragraphs: what technology shifts are happening, what the market is betting on for 3-5 years, why investors assign high/low valuations based on future potential>",
  "market_context": "<2-3 paragraphs: supply/demand dynamics, competition evolution, who is gaining/losing share and why, major capex or strategic bets being made>",
  "geopolitical_context": "<2-3 paragraphs: specific policies (name + date), trade actions, export controls, subsidies, geographic risks, which companies benefit vs. suffer — with numbers where found>",
  "expert_sentiment": "<what specific analysts, executives, and investors are saying — quote firms, price targets, themes from earnings calls>",
  "key_risks": ["<specific risk with mechanism>", "<risk 2>", "<risk 3>", "<risk 4>"],
  "qualitative_insights": [
    "<forward-looking insight: what the market is pricing in / betting on>",
    "<competitive dynamic not visible in financial metrics>",
    "<geopolitical or policy factor shaping next 2-3 years>"
  ],
  "sources_consulted": [
    {"title": "<article/report title>", "url": "<url>", "snippet": "<key excerpt or data point found>"},
    ...
  ]
}
"""


def _build_researcher_prompt() -> str:
    today = date.today().isoformat()
    header = (
        f"You are the Research agent in a multi-agent Sector Analyst system.\n"
        f"Today's date: {today}\n\n"
        f"RECENCY REQUIREMENT: You are researching as of {today}. Include the current year\n"
        f"(and prior year where relevant) in every search query to surface the most recent\n"
        f"results. Prioritize sources from 2025 and 2026. Do not rely on generic queries\n"
        f"that may return stale 2023-2024 content.\n\n"
    )
    return header + _RESEARCHER_PROMPT_BODY


@function_tool(strict_mode=False)
def search_web(query: str, max_results: int = 8) -> list[dict]:
    """
    Search the web using DuckDuckGo for recent news, analyst reports, and articles.

    Args:
        query: Search query string (be specific — include sector name, company names, or topics)
        max_results: Number of results to return (default 8, max 10)

    Returns:
        List of results with 'title', 'url', 'snippet' fields
    """
    results = _search_web(query, max_results=min(max_results, 10))
    # Filter out empty results
    return [r for r in results if r.get("title") and r.get("url")]


@function_tool(strict_mode=False)
def fetch_page_text(url: str, max_chars: int = 4000) -> dict:
    """
    Fetch and extract the main text content from a web page.

    Use this on the most promising search results to read the full article or report.

    Args:
        url: The full URL to fetch
        max_chars: Maximum characters to return (default 4000)

    Returns:
        Dict with 'url', 'title', 'text', 'char_count' fields
    """
    return _fetch_page_text(url, max_chars=max_chars)


def build_researcher_agent() -> Agent:
    return Agent(
        name="Researcher",
        model=_make_model(),
        instructions=_build_researcher_prompt(),
        tools=[search_web, fetch_page_text],
    )
