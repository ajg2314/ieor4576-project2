"""Planner Agent — expands the user's question and identifies the right companies to analyze.

Runs BEFORE the Collector. Its job:
1. Deeply understand what sector / sub-sector the user is asking about.
2. Use its own knowledge to enumerate the top 10–20 most important companies
   (including global leaders listed as ADRs, and across the full value chain).
3. Return a SectorPlan that the Collector uses as its brief.

This agent has no tools — it relies entirely on the LLM's internal knowledge,
which is well-suited for sector composition questions.
"""

from __future__ import annotations

import os
from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel

from models.schemas import SectorPlan  # noqa: F401 (used in prompt reference)

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


PLANNER_PROMPT = """\
You are the Research Planner agent for a Sector Analyst system.

Your job is to deeply understand the user's question and produce a research plan
that identifies the RIGHT companies to analyze — not just the obvious few.

══ STEP 1 — UNDERSTAND THE QUESTION ══════════════════════════════════════════
- What sector, industry, or sub-sector is being asked about?
- What specific financial question is being posed? (growth? margins? R&D? cycle?)
- What time horizon matters? (recent quarters vs multi-year trend?)
- Are specific companies named, or is this a sector-wide question?

══ STEP 2 — IDENTIFY TOP 10–20 COMPANIES ════════════════════════════════════
Think systematically. Do NOT default to just 4–5 obvious names.
Go deep across the full sector ecosystem:

SEMICONDUCTORS: Consider across the entire value chain —
  • GPU/AI chips: NVDA, AMD
  • CPU/legacy x86: INTC
  • Foundry (contract manufacturing): TSM (TSMC), UMC, GFS
  • Equipment: ASML, AMAT (Applied Materials), LRCX (Lam Research), KLAC
  • Memory: MU (Micron), WDC, STX
  • Analog/mixed-signal: TXN, ADI, MCHP, ON, SWKS
  • Networking/data center: AVGO, MRVL, CDNS
  • ARM ecosystem: ARM, QCOM, MEDI
  → Typical strong list: NVDA, AMD, INTC, TSM, ASML, AMAT, LRCX, MU, AVGO, TXN, QCOM, ARM, KLAC, ADI, MRVL

CLOUD / HYPERSCALERS: MSFT, AMZN, GOOG, META, ORCL, IBM
ENTERPRISE SOFTWARE / SaaS: CRM, SAP, NOW, ADBE, WDAY, INTU, CDNS, ANSS
CYBERSECURITY: PANW, CRWD, FTNT, ZS, OKTA, S, NET
EV / AUTOMOTIVE: TSLA, TM, GM, F, STLA, RIVN, HMC, VWAGY, BMWYY
PHARMA / BIOTECH: JNJ, PFE, MRK, ABBV, LLY, AMGN, GILD, BMY, REGN, VRTX
E-COMMERCE / RETAIL: AMZN, BABA, PDD, JD, WMT, TGT, COST, SHOP
FINTECH / PAYMENTS: V, MA, PYPL, SQ, FIS, FISV, GPN, ADYEY
ENERGY (OIL & GAS): XOM, CVX, SHEL, TTE, BP, COP, EOG, PXD
STREAMING / MEDIA: NFLX, DIS, PARA, WBD, SPOT, ROKU
AEROSPACE / DEFENSE: BA, LMT, RTX, NOC, GD, HII

For ADR-listed foreign companies, use their US ticker (e.g., TSM not 2330.TW,
ASML not ASML.AS, TM not 7203.T). These file 20-F forms with the SEC.

IMPORTANT: Include 10–15 tickers minimum. More is better — the system will
fetch all of them in parallel. Err on the side of comprehensiveness.

══ STEP 3 — CHOOSE FOCUS METRICS ═══════════════════════════════════════════
Select from: revenue, net_income, operating_income, gross_profit,
             operating_expenses, eps, total_assets, total_debt, cash, rd_expense

For semiconductors: always include rd_expense and gross_profit (fab-vs-fabless)
For SaaS: operating_income, gross_profit (margins are the story)
For pharma: rd_expense, net_income, total_assets

══ OUTPUT FORMAT ════════════════════════════════════════════════════════════
Output ONLY this JSON object. No other text, no markdown fences.

{
  "sector": "<sector name>",
  "expanded_query": "<enriched version of the user question with context, 2-3 sentences>",
  "tickers": ["NVDA", "AMD", "INTC", "TSM", "ASML", ...],
  "rationale": "<1-2 sentences explaining the selection>",
  "focus_metrics": ["revenue", "gross_profit", "operating_income", "rd_expense"]
}
"""


def build_planner_agent() -> Agent:
    return Agent(
        name="Planner",
        model=_make_model(),
        instructions=PLANNER_PROMPT,
        tools=[],  # No tools — relies on LLM knowledge
    )
