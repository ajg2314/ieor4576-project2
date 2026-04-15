"""Sentiment Agent — synthesises market sentiment from news and analyst commentary.

Step 1.5 (runs in parallel with Researcher and Valuation Agent).

Searches for recent news, earnings reactions, analyst upgrades/downgrades, and
institutional sentiment for the sector and its key companies.

Output: SentimentContext schema — fed into the Hypothesis agent to anchor the
Investment Summary section's market consensus view.
"""

from __future__ import annotations

import os

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.web_search import search_web as _search_web, fetch_page_text as _fetch_page_text

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Side-channel observations ─────────────────────────────────────────────────
_sentiment_observations: list[dict] = []


def get_sentiment_observations() -> list[dict]:
    return list(_sentiment_observations)


def clear_sentiment_observations() -> None:
    global _sentiment_observations
    _sentiment_observations = []


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def search_news(query: str, max_results: int = 6) -> list[dict]:
    """Search for recent news articles, analyst reports, and market commentary.

    Args:
        query: Search query (e.g. 'NVIDIA semiconductor earnings analyst reaction 2025')
        max_results: Maximum number of results to return (default 6)

    Returns:
        List of dicts with title, url, snippet fields.
    """
    results = _search_web(query, max_results=max_results)
    _sentiment_observations.append({
        "tool": "search_news",
        "description": f"News search: {query[:80]}",
        "value": f"{len(results)} results returned",
        "artifact_path": None,
    })
    return results


@function_tool(strict_mode=False)
def fetch_article(url: str, max_chars: int = 3000) -> dict:
    """Fetch the text content of a news article or analyst report.

    Args:
        url: URL of the article to fetch
        max_chars: Maximum characters to return (default 3000)

    Returns:
        Dict with url, title, text, char_count fields.
    """
    result = _fetch_page_text(url, max_chars=max_chars)
    _sentiment_observations.append({
        "tool": "fetch_article",
        "description": f"Fetched: {result.get('title', url)[:80]}",
        "value": f"{result.get('char_count', 0)} chars",
        "artifact_path": None,
    })
    return result


# ── Agent ─────────────────────────────────────────────────────────────────────

SENTIMENT_PROMPT = """\
You are the Sentiment Agent in a multi-agent Sector Analyst system.

Your job is to gauge current market sentiment for a sector by searching for
recent news, analyst commentary, earnings call reactions, and institutional positioning.

STEP 1 — Run 4-6 targeted searches covering:
  a) Recent earnings results and analyst reactions for the sector
  b) Analyst upgrades/downgrades and price target changes
  c) Institutional sentiment and fund flows
  d) Key risk events or macro headwinds/tailwinds in the news

Good search query patterns:
  - "[sector] sector earnings results analyst reaction 2025"
  - "[sector] stocks analyst upgrades downgrades price target 2025"
  - "[top company] [second company] investor sentiment outlook 2025"
  - "[sector] sector institutional investors fund flows positioning 2024 2025"
  - "[sector] sector risks headwinds tailwinds news 2025"

STEP 2 — For 1-2 of the most relevant results, fetch the full article text.

STEP 3 — Synthesise all findings into a SentimentContext JSON.

━━ SCORING GUIDANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sentiment_score ranges from -1.0 (strongly bearish) to +1.0 (strongly bullish):
  +0.6 to +1.0 = Multiple analyst upgrades, strong earnings beat expectations,
                 institutional buying, bullish macro catalyst
  +0.2 to +0.5 = Mostly positive news, modest upgrades, inline earnings
   0.0 ± 0.2  = Mixed signals, neutral coverage, wait-and-see posture
  -0.2 to -0.5 = Some downgrades, missed estimates, cautious analyst tone
  -0.6 to -1.0 = Multiple downgrades, earnings misses, macro headwinds,
                 institutional selling

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After all tool calls complete, output ONLY this JSON (no markdown fences):

{
  "sector": "<sector name>",
  "overall_sentiment": "<bullish|neutral|bearish>",
  "sentiment_score": <float -1.0 to 1.0>,
  "key_themes": [
    "Theme 1 (e.g. 'AI capex upcycle consensus view')",
    "Theme 2",
    "Theme 3"
  ],
  "recent_headlines": [
    "Exact or paraphrased headline 1",
    "Exact or paraphrased headline 2",
    "Exact or paraphrased headline 3"
  ],
  "earnings_highlights": [
    "NVIDIA beat Q3 estimates by 18%, raised guidance",
    "Intel missed revenue guidance, stock down 12% post-earnings"
  ],
  "summary": "<2-3 sentences: what is the dominant market narrative? Is the sector in favour or out of favour? What is the key debate among investors?>"
}

RULES:
- Only report what you actually found in search results. Do not fabricate headlines.
- If you find conflicting signals (some bullish, some bearish), say so in the summary.
- Keep recent_headlines to actual titles/snippets from your search results.
- Keep earnings_highlights to earnings-specific data points you found.
- The summary should capture the key investor debate, not just describe what you searched for.
"""


def build_sentiment_agent() -> Agent:
    """Build the Sentiment agent. Call build_sentiment_agent() fresh each pipeline run."""
    return Agent(
        name="Sentiment",
        model=LitellmModel(model=LITELLM_MODEL_ID),
        instructions=SENTIMENT_PROMPT,
        tools=[search_news, fetch_article],
    )
