"""Orchestrator Agent — plans the analysis and hands off to specialist agents.

Multi-agent pattern: Orchestrator-handoff. The orchestrator decides whether to
collect more data, re-run EDA, or finalize the hypothesis.
"""

from __future__ import annotations

import os
from agents import Agent, Runner, handoff
from agents.extensions.models.litellm_model import LitellmModel

from .collector import build_collector_agent
from .eda_agent import build_eda_agent
from .hypothesis_agent import build_hypothesis_agent

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

LITELLM_MODEL_ID = f"vertex_ai/{GEMINI_MODEL}"


def _make_model() -> LitellmModel:
    return LitellmModel(model=LITELLM_MODEL_ID)


ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of a multi-agent Sector Analyst system.

The user will ask you to analyze a sector, compare companies, or investigate
a financial trend. Your job is to coordinate the full pipeline:

  Collect → Explore → Hypothesize

PIPELINE:

1. Hand off to the Collector agent.
   Pass: the user's original question, the sector or companies to analyze.
   The Collector retrieves SEC EDGAR financial data and filing text.

2. Hand off to the EDA agent.
   Pass: the DataBundle from the Collector + the user's question.
   The EDA agent computes financial metrics and generates charts.

3. Evaluate the EDA findings.
   - If the EDA reveals data gaps (e.g., a company is missing revenue data,
     or only 2 years of history), loop back to the Collector with a more
     specific request (different concepts or additional tickers).
   - If findings are solid (specific numbers, clear trends), proceed.

4. Hand off to the Hypothesis agent.
   Pass: user question + DataBundle + EDAFindings.
   The Hypothesis agent produces the final grounded report.

Rules:
- Never answer the analytics question yourself. Always delegate.
- When handing off, always include the original user question as context.
- You may iterate (Collect → EDA → Collect → EDA) at most twice before
  forcing a hypothesis from available data.
- Keep track of what data has been collected to avoid redundant requests.
"""


def build_orchestrator() -> Agent:
    collector = build_collector_agent()
    eda = build_eda_agent()
    hypothesis = build_hypothesis_agent()

    return Agent(
        name="Orchestrator",
        model=_make_model(),
        instructions=ORCHESTRATOR_PROMPT,
        handoffs=[
            handoff(collector),
            handoff(eda),
            handoff(hypothesis),
        ],
    )


async def run_analysis(question: str) -> str:
    """Entry point: run the full multi-agent pipeline for a user question."""
    orchestrator = build_orchestrator()
    result = await Runner.run(orchestrator, input=question)
    return result.final_output
