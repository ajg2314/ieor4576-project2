"""Sector Specialist — parameterized domain expert agent for technology and competitive analysis.

Routes to one of 5 specialist types based on the sector name:
  tech         → semiconductors, cloud, software, AI, hardware
  biomedical   → pharma, biotech, medical devices, healthcare
  energy       → oil & gas, renewables, utilities, materials, mining
  financials   → banks, insurance, asset management, fintech
  general      → any other sector (default fallback)

Runs in parallel with Researcher, Valuation, Sentiment, and Geopolitical Advisor.
Also callable as a tool from the Hypothesis agent for targeted follow-up questions.

Produces SectorAnalysis: SOTA technology, competitive dynamics, forward thesis,
and a ready-to-paste summary for report Section 4.
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
_specialist_observations: list[dict] = []


def get_specialist_observations() -> list[dict]:
    return list(_specialist_observations)


def clear_specialist_observations() -> None:
    global _specialist_observations
    _specialist_observations = []


# ── Sector routing ────────────────────────────────────────────────────────────

def _route_specialist_type(sector: str) -> str:
    """Map a sector name string to one of 5 specialist types."""
    s = sector.lower()
    if any(w in s for w in ["pharma", "bio", "health", "medical", "drug", "therapeut"]):
        return "biomedical"
    if any(w in s for w in ["semi", "chip", "tech", "software", "cloud", "ai", "hardware", "compute"]):
        return "tech"
    if any(w in s for w in ["energy", "oil", "gas", "renew", "material", "mining", "chem", "util", "lithium"]):
        return "energy"
    if any(w in s for w in ["bank", "financ", "insur", "asset", "invest", "capital", "credit"]):
        return "financials"
    return "general"


# ── Per-specialist prompt fragments ───────────────────────────────────────────

_SPECIALIST_FOCUS = {
    "tech": """\
You are a Technology & Semiconductor Specialist with deep expertise in:
- Compute architectures: GPU vs. CPU vs. ASIC, AI training vs. inference workloads
- Software moats: CUDA ecosystem, cloud platform lock-in, developer toolchains
- Process nodes: leading-edge (3nm/2nm), EUV lithography, advanced packaging (CoWoS, chiplets)
- AI/ML SOTA: transformer architectures, model scaling laws, inference optimization
- Supply chain: fabless/foundry/OSAT ecosystem, HBM memory, advanced packaging bottlenecks

RESEARCH FOCUS for your web searches:
- Latest AI chip launches and benchmarks (NVIDIA, AMD, Google TPU, Amazon Trainium)
- Process node competition (TSMC vs. Samsung vs. Intel Foundry)
- AI model scaling trends and what hardware they demand
- Custom silicon programs at hyperscalers (Apple M-series, Google TPU, Amazon Graviton)
- CUDA alternatives: ROCm, oneAPI, OpenCL — are they gaining traction?
- Edge AI inference chips (Qualcomm NPU, Apple Neural Engine)
""",

    "biomedical": """\
You are a Biomedical & Pharmaceutical Specialist with deep expertise in:
- Drug development pipeline: clinical trial phases, FDA approval process, PDUFA dates
- Therapeutic modalities: small molecules, biologics, mRNA, CRISPR, CAR-T, ADCs
- GLP-1 receptor agonists (Ozempic, Wegovy, Mounjaro): mechanism, market penetration, pipeline
- Patent cliffs and biosimilar dynamics
- Healthcare pricing: PBM negotiations, Medicare drug price negotiation (IRA provisions)
- Medical devices: FDA clearance pathways (510k vs PMA), robotic surgery, continuous monitoring

RESEARCH FOCUS for your web searches:
- Recent FDA approvals and PDUFA decisions (2024-2025)
- GLP-1 market competition: oral formulations, next-gen drugs in Phase 3
- CRISPR and gene therapy clinical readouts
- Major patent expirations and biosimilar entrants
- Healthcare AI: drug discovery acceleration, clinical trial optimization
- Hospital system purchasing trends and device adoption rates
""",

    "energy": """\
You are an Energy & Materials Specialist with deep expertise in:
- Upstream oil & gas: E&P economics, shale breakeven costs, OPEC+ supply dynamics
- LNG: Henry Hub vs. JKM spread, US export terminal capacity, European demand
- Energy transition: solar/wind LCOE curves, grid storage, green hydrogen economics
- IRA tax credits: PTC, ITC, clean hydrogen, advanced manufacturing credits
- Critical minerals: lithium, cobalt, nickel, rare earths — supply chain and pricing
- Commodity cycles: how macro environment, inventory cycles, and capex decisions drive prices

RESEARCH FOCUS for your web searches:
- Current oil price drivers and OPEC+ compliance data
- LNG export volumes and contract pricing
- Battery technology advances (solid-state, sodium-ion alternatives to lithium-ion)
- IRA implementation: which projects are getting funded, which credits are being claimed
- Critical mineral supply chain investments (US DoD, Australian Critical Minerals Strategy)
- Renewable energy capacity additions and grid integration challenges
""",

    "financials": """\
You are a Financial Services Specialist with deep expertise in:
- Bank profitability: NIM dynamics, deposit repricing, loan growth, credit quality
- Capital adequacy: Basel III/IV, CET1 ratios, stress test results (Fed DFAST)
- Credit cycle: commercial real estate stress, consumer credit quality, leverage loan markets
- Insurance: combined ratio trends, catastrophe loss modeling, reserve adequacy
- Asset management: fee compression, active-to-passive shift, alternatives growth
- Fintech disruption: payment networks, embedded finance, digital banking competition

RESEARCH FOCUS for your web searches:
- Federal Reserve rate policy expectations and their impact on bank NIM
- Commercial real estate (CRE) loan losses — office sector specifically
- Bank earnings themes from most recent quarter (Q4 2024 / Q1 2025)
- Basel III Endgame final rule status and capital requirement changes
- Insurance catastrophe losses (wildfire, hurricane) and reinsurance pricing
- Asset management flows: where is money going (active vs passive, alternatives)?
""",

    "general": """\
You are a Sector Analysis Specialist with broad expertise in competitive dynamics,
market structure, barriers to entry, and industry evolution across all sectors.

RESEARCH FOCUS for your web searches:
- Market structure: concentration ratios, barriers to entry, pricing power
- Competitive dynamics: who is gaining share and why, M&A activity
- Regulatory environment: antitrust, environmental, safety regulations
- International competition: where is global competition coming from?
- Technology disruption: which technologies could change the competitive landscape?
- Supply chain: where are the dependencies and concentration risks?
""",
}


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def search_sector_research(query: str, max_results: int = 6) -> list[dict]:
    """Search for recent research, news, and analysis relevant to the sector.

    Args:
        query: Search query focused on sector technology, competition, or SOTA
        max_results: Maximum results to return (default 6)

    Returns:
        List of dicts with title, url, snippet fields.
    """
    results = _search_web(query, max_results=max_results)
    _specialist_observations.append({
        "tool": "search_sector_research",
        "description": f"Sector search: {query[:80]}",
        "value": f"{len(results)} results",
        "artifact_path": None,
    })
    return results


@function_tool(strict_mode=False)
def fetch_article(url: str, max_chars: int = 3000) -> dict:
    """Fetch the full text of an article, research paper, or technical report.

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default 3000)

    Returns:
        Dict with url, title, text, char_count fields.
    """
    result = _fetch_page_text(url, max_chars=max_chars)
    _specialist_observations.append({
        "tool": "fetch_article",
        "description": f"Fetched: {result.get('title', url)[:80]}",
        "value": f"{result.get('char_count', 0)} chars",
        "artifact_path": None,
    })
    return result


@function_tool(strict_mode=False)
def get_domain_knowledge(sector: str, topic: str) -> str:
    """Retrieve domain knowledge about the sector from the knowledge base.

    Args:
        sector: Sector being analyzed (e.g. 'semiconductors', 'biomedical', 'energy')
        topic: What you want to know (e.g. 'business models and revenue drivers',
               'key metrics to compute', 'technology trends', 'competitive dynamics')

    Returns:
        Relevant knowledge chunks from the sector knowledge base.
    """
    result = retrieve_sector_knowledge(sector, topic, n=4)
    if not result:
        result = retrieve_sector_knowledge("general", topic, n=3)
    _specialist_observations.append({
        "tool": "get_domain_knowledge",
        "description": f"RAG: {sector} / {topic[:60]}",
        "value": f"{len(result)} chars retrieved",
        "artifact_path": None,
    })
    return result or "No knowledge base entries found for this topic."


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_specialist_prompt(specialist_type: str, sector: str) -> str:
    today = date.today().isoformat()
    focus = _SPECIALIST_FOCUS.get(specialist_type, _SPECIALIST_FOCUS["general"])
    date_header = (
        f"Today's date: {today}\n\n"
        f"RECENCY REQUIREMENT: You are analyzing as of {today}. Include the current year\n"
        f"in every search query (e.g. 'AI chip trends 2025 2026', 'biotech FDA approvals 2025').\n"
        f"Prioritize SOTA developments, competitive moves, and market data from 2025 and 2026.\n\n"
    )
    return date_header + f"""\
{focus}

Your sector for this analysis: {sector}

━━ RESEARCH PROCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — ALWAYS call get_domain_knowledge(sector, topic) FIRST for at least 2 different
         topics (e.g. "technology trends", "competitive dynamics"). This always works and
         gives a solid knowledge-base foundation.

STEP 2 — Then call search_sector_research() to supplement with current events (4-6 queries):
  - Current SOTA and recent technology advances
  - Who is winning competitively and what structural advantages they hold
  - What the market expects over the next 3-5 years (earnings call themes, analyst reports)
  - Key disruptions that could reshape the competitive landscape
  - Forward-looking investment thesis: what are long-term investors betting on?
  If search returns an empty list ([]), skip to STEP 3 — do NOT report the failure.

STEP 3 — If web search returned results, fetch 1-2 most informative articles.
         If web search was unavailable, call get_domain_knowledge() with 2 additional topics
         to fill gaps. Use your training knowledge to supplement.

STEP 4 — Output SectorAnalysis JSON.

━━ OUTPUT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Always populate ALL JSON fields with substantive content — never leave a field empty.
- If web search was unavailable, still write technology_sota, competitive_dynamics,
  forward_thesis using knowledge-base results + your training knowledge.
- Append to the summary field ONLY IF search completely failed:
  " [Note: real-time web search unavailable — analysis reflects knowledge base only. Re-run for latest data.]"
- NEVER write a confession of tool failure into any field.
- NEVER leave any field blank or write "Unable to research" or similar.

━━ ANALYSIS RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORWARD-LOOKING is the priority — not what happened, but what the market expects:
- The hypothesis agent already has historical financial data from SEC filings
- Your unique value is domain expertise about TECHNOLOGY and COMPETITIVE DYNAMICS
- Focus on: which companies have durable moats, which are being disrupted, and WHY

DEPTH over breadth:
- 2-3 deep, specific insights beat 10 generic observations
- Name specific products, research papers, or competitive moves, not just general trends
- Quantify where you can: market sizes, TAM estimates, adoption rates, cost curves

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After all tool calls, output ONLY this JSON (no markdown fences):

{{
  "sector": "{sector}",
  "specialist_type": "{specialist_type}",
  "technology_sota": "<2-3 paragraphs on current state of the art: what is technically possible today, what recent breakthroughs matter, what is in labs that will matter in 2-3 years>",
  "competitive_dynamics": "<2-3 paragraphs on who holds durable competitive advantages and why: software moats, network effects, manufacturing scale, regulatory capture, proprietary data — and who is most vulnerable to disruption>",
  "forward_thesis": "<2 paragraphs on what long-term investors are betting on: what scenario must come true for the sector leaders to justify their current valuations? What are the key assumptions, and what evidence supports or challenges them?>",
  "key_disruptions": [
    "Specific emerging threat or opportunity 1 with mechanism",
    "Specific emerging threat or opportunity 2",
    "Specific emerging threat or opportunity 3"
  ],
  "summary": "<2-3 paragraphs synthesizing SOTA + competitive dynamics + forward thesis, written at the level of a CFA-trained analyst, ready to paste directly into a professional analyst report's Technology & Innovation section>"
}}
"""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_sector_specialist_agent(sector: str) -> Agent:
    """Build a sector specialist agent for the given sector. Call fresh each pipeline run.

    Args:
        sector: Sector name from SectorPlan (e.g. 'Semiconductors', 'Cloud Software')

    Returns:
        Agent configured as the appropriate specialist type.
    """
    specialist_type = _route_specialist_type(sector)
    prompt = _build_specialist_prompt(specialist_type, sector)
    return Agent(
        name=f"SectorSpecialist_{specialist_type}",
        model=LitellmModel(model=LITELLM_MODEL_ID),
        instructions=prompt,
        tools=[search_sector_research, fetch_article, get_domain_knowledge],
    )
