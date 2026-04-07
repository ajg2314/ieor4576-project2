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
    return LitellmModel(
        model=LITELLM_MODEL_ID,
        api_base=None,
        extra_kwargs={
            "vertex_project": VERTEX_PROJECT,
            "vertex_location": VERTEX_LOCATION,
        },
    )


ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of a multi-agent data analysis system.

Your job is to receive a user's analytics question and coordinate the full
Collect → Explore → Hypothesize pipeline:

1. Hand off to the Collector agent to retrieve relevant real-world data.
2. Hand off to the EDA agent to explore the data and surface key findings.
3. If the EDA reveals gaps or raises new questions, loop back to the Collector
   for additional data (iterative refinement).
4. Once EDA findings are solid, hand off to the Hypothesis agent to produce
   the final report.

Rules:
- Never answer the analytics question yourself. Always delegate.
- After each handoff returns, evaluate whether the result is sufficient.
  If data is missing or ambiguous, initiate another collection round.
- Pass the full context (question + prior findings) when handing off.
- Stop iterating when you have sufficient evidence for a grounded hypothesis.
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
