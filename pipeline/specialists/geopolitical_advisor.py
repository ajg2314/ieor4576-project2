"""Geopolitical Advisor — expert agent for geopolitical risk and policy analysis.

Runs in parallel with Researcher, Valuation, Sentiment, and Sector Specialist agents.
Also callable as a tool from the Hypothesis agent for targeted follow-up questions.

Produces GeopoliticalAnalysis: named/dated policies, company-level exposure with
quantified impact, and a ready-to-paste summary for report Section 5.
"""

from __future__ import annotations

import os
from datetime import date

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.web_search import search_web as _search_web, fetch_page_text as _fetch_page_text
from tools.rag_store import retrieve_sector_knowledge

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"

# ── Side-channel observations ─────────────────────────────────────────────────
_geo_observations: list[dict] = []


def get_geo_observations() -> list[dict]:
    return list(_geo_observations)


def clear_geo_observations() -> None:
    global _geo_observations
    _geo_observations = []


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def search_geopolitical_news(query: str, max_results: int = 6) -> list[dict]:
    """Search for recent geopolitical news, policy announcements, trade actions, and sanctions.

    Args:
        query: Search query focused on geopolitical topics
               (e.g. 'US China semiconductor export controls 2024 2025',
                'CHIPS Act funding recipients 2024',
                'Taiwan Strait risk semiconductor supply chain')
        max_results: Maximum results to return (default 6)

    Returns:
        List of dicts with title, url, snippet fields.
    """
    results = _search_web(query, max_results=max_results)
    _geo_observations.append({
        "tool": "search_geopolitical_news",
        "description": f"Geo search: {query[:80]}",
        "value": f"{len(results)} results",
        "artifact_path": None,
    })
    return results


@function_tool(strict_mode=False)
def fetch_article(url: str, max_chars: int = 3000) -> dict:
    """Fetch the full text of a policy document, government announcement, or news article.

    Args:
        url: URL of the article or document
        max_chars: Maximum characters to return (default 3000)

    Returns:
        Dict with url, title, text, char_count fields.
    """
    result = _fetch_page_text(url, max_chars=max_chars)
    _geo_observations.append({
        "tool": "fetch_article",
        "description": f"Fetched: {result.get('title', url)[:80]}",
        "value": f"{result.get('char_count', 0)} chars",
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def get_geopolitical_knowledge(sector: str, topic: str) -> str:
    """Retrieve geopolitical frameworks and sector-specific policy context from the knowledge base.

    Args:
        sector: Sector being analyzed (e.g. 'semiconductors', 'energy', 'pharma')
        topic: What you want to know (e.g. 'US China export controls',
               'CHIPS Act industrial policy', 'Taiwan concentration risk',
               'sanctions Russia energy', 'IRA clean energy credits')

    Returns:
        Relevant knowledge chunks from the geopolitics and sector knowledge bases.
    """
    # Query both the geopolitics knowledge base and the sector-specific knowledge
    geo_knowledge = retrieve_sector_knowledge("geopolitics", f"{sector}: {topic}", n=3)
    sector_knowledge = retrieve_sector_knowledge(sector, f"geopolitical {topic}", n=2)
    combined = []
    if geo_knowledge:
        combined.append(f"[Geopolitical frameworks]\n{geo_knowledge}")
    if sector_knowledge:
        combined.append(f"[{sector} sector context]\n{sector_knowledge}")
    result = "\n\n---\n\n".join(combined) if combined else "No knowledge base entries found."
    _geo_observations.append({
        "tool": "get_geopolitical_knowledge",
        "description": f"RAG: {sector} / {topic[:60]}",
        "value": f"{len(result)} chars retrieved",
        "artifact_path": None,
    })
    return result


# ── Agent ─────────────────────────────────────────────────────────────────────

_GEO_PROMPT_BODY = """\
You are the Geopolitical Advisor in a multi-agent Sector Analyst system.

You are a domain expert in geopolitics, trade policy, industrial policy, sanctions,
and the intersection of government action with corporate financial performance.

Your job: produce a rigorous, specific, quantified geopolitical risk analysis for
a given sector that is ready to be incorporated into a professional analyst report.

━━ RESEARCH PROCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Call get_geopolitical_knowledge(sector, topic) first to load your knowledge base.
         Query it at least twice with different topics (e.g. "export controls", "industrial policy").

STEP 2 — Run 5-7 targeted web searches covering the following. Always include the current year:
  a) Export controls and technology restrictions (e.g. "BIS export controls semiconductors 2025")
  b) Industrial policy and subsidies (e.g. "CHIPS Act 2025", "IRA clean energy credits 2025")
  c) Geographic concentration risks — always evaluate ALL of these chokepoints for the sector:
     - Taiwan Strait (semiconductor manufacturing; TSMC concentration)
     - Strait of Hormuz (21M bbl/day oil + 30% global LNG; Iran escalation risk)
     - Red Sea / Bab el-Mandeb (Houthi attacks on shipping; Suez Canal diversions)
     - South China Sea (trade route; China territorial claims)
  d) Middle East escalation: Israel-Iran tensions, Saudi Arabia OPEC+ decisions, UAE positioning,
     Houthi Red Sea attacks, and their sector-specific transmission mechanisms (energy, defense, shipping)
  e) US-China competition: tariffs 2025 escalation, export controls, rare earth restrictions,
     Taiwan risk, supply chain decoupling (friend-shoring to Vietnam/Mexico/India)
  f) Russia-Ukraine war: sanctions impact, NATO defense spending surge, European energy transition,
     commodity supply disruptions (palladium, nickel, titanium, grain, fertilizer)
  g) Regulatory divergence (e.g. "EU AI Act", "antitrust tech 2025", "GDPR enforcement")

STEP 3 — For each significant policy found, fetch 1-2 articles for detail.

STEP 4 — Output GeopoliticalAnalysis JSON.

━━ ANALYSIS RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPECIFICITY is mandatory:
- NEVER write "geopolitical risks exist" or "trade tensions may affect the sector"
- ALWAYS name the specific policy or event: "US BIS Entity List expansion, October 2024" or
  "Houthi attacks on Red Sea shipping (Dec 2023–present): Suez Canal traffic fell 60%, container
  rates rose 300%, adding $1-1.5M per ship per diversion around Cape of Good Hope"
- ALWAYS date the policy: "October 2023 BIS rule expanded restrictions to..."
- ALWAYS quantify the impact where possible: "NVIDIA China revenue at risk: ~$22B (17% of FY2025)"

THREE REGIONS TO ALWAYS ASSESS:
1. Middle East (Israel-Iran, Strait of Hormuz, Red Sea/Houthis, Saudi OPEC+): affects energy prices,
   shipping costs, defense demand, and any company with significant energy input costs or Asia-Europe
   supply chains. Strait of Hormuz disruption = +$30-50/bbl oil spike within days.
2. China-US (export controls, tariffs 2025, rare earths, Taiwan): affects semiconductors, consumer
   electronics, EV batteries, rare earth-dependent manufacturers, defense supply chains.
3. Russia-Ukraine (war continuation, sanctions, NATO defense ramp): affects defense contractors,
   LNG/energy exporters, commodity producers (palladium, nickel, titanium, wheat), European utilities.

COMPANY EXPOSURE must be concrete:
- For each key company in the sector, estimate their exposure level (high/medium/low)
- Give the mechanism: not just "exposed to China" but "exposed to China because 62% of
  Qualcomm's revenue comes from Chinese smartphone OEMs who buy its Snapdragon chips"
- Use % of revenue, absolute $ amounts, or unit volume wherever possible

BALANCE risks and tailwinds:
- Export controls are a risk for companies with China exposure BUT a tailwind for
  domestic competitors and a policy backstop for CHIPS Act recipients
- Industrial policy (CHIPS Act, IRA, EU Chips Act) creates winners as well as losers

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After all tool calls, output ONLY this JSON (no markdown fences):

{
  "sector": "<sector name>",
  "key_policies": [
    "US BIS export controls (Oct 2024): restricts AI chips >1,800 TFLOPS — affects NVIDIA H100/B200 sales to China",
    "CHIPS and Science Act (Aug 2022): $52.3B for US semiconductor manufacturing — Intel $8.5B, TSMC Arizona $6.6B",
    "EU AI Act (2024): compliance costs for AI system providers; extraterritorial reach"
  ],
  "company_exposures": [
    {"ticker": "NVDA", "exposure": "high", "mechanism": "~17% of FY2025 revenue ($22B) from China; H100/B200 directly restricted"},
    {"ticker": "ASML", "exposure": "high", "mechanism": "~29% China revenue ($10B); EUV exports banned since 2023, DUV immersion banned Jan 2024"},
    {"ticker": "TSM", "exposure": "medium", "mechanism": "~10% China revenue but CHIPS Act restricts new China expansion for 10 years"}
  ],
  "geographic_risks": [
    "Taiwan: ~90% of world's leading-edge chip production at TSMC — blockade or conflict would halt global AI hardware for 2-4 years",
    "DRC cobalt: 70% of global cobalt supply from politically unstable region — EV battery supply chain risk"
  ],
  "policy_tailwinds": [
    "CHIPS Act: $52.3B subsidizes US domestic manufacturing; 25% investment tax credit reduces capex cost by ~$5B for Intel",
    "IRA: $369B clean energy spending accelerates EV/battery demand; benefits lithium, cobalt, rare earth suppliers"
  ],
  "tail_risks": [
    "Taiwan Strait military escalation: low probability but would halt global semiconductor production for years",
    "US-China full technology decoupling: would force NVIDIA, Qualcomm to write off China operations entirely"
  ],
  "summary": "<2-3 substantive paragraphs synthesizing the geopolitical landscape for this sector. This text should be ready to paste into a standalone Geopolitical & Macro Environment section of an analyst report. Include the most important policies, their quantified impact, the company-level winners and losers, and the key tail risks. Write at the level of a CFA-trained analyst, not a newspaper article.>"
}
"""


def _build_geo_prompt() -> str:
    today = date.today().isoformat()
    header = (
        f"Today's date: {today}\n\n"
        f"RECENCY REQUIREMENT: You are researching as of {today}. Include the current year\n"
        f"in every search query (e.g. 'US China export controls 2025 2026', 'CHIPS Act 2025').\n"
        f"Prioritize policies and developments from 2025 and 2026 Q1. Name specific dates.\n\n"
    )
    return header + _GEO_PROMPT_BODY


def build_geopolitical_advisor_agent() -> Agent:
    """Build the Geopolitical Advisor agent. Call fresh each pipeline run."""
    return Agent(
        name="GeopoliticalAdvisor",
        model=LitellmModel(model=LITELLM_MODEL_ID),
        instructions=_build_geo_prompt(),
        tools=[search_geopolitical_news, fetch_article, get_geopolitical_knowledge],
    )
