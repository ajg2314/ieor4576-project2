"""Q&A Agent — interactive follow-up and summarization for generated analyst reports.

Two operating modes:
  summarize          → produces a concise 3-page executive summary of the full report
  ask: <question>    → answers a specific follow-up question using the report as context

The full report text and specialist analyses are injected into the agent's instructions
at build time so the model treats them as persistent context (not a one-shot message).
Gemini 2.5 Flash has a 1M token context window — large reports are not a problem.

The agent can call search_for_more_context() for questions requiring information
outside the report, and consult_specialist() for deep domain follow-ups.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.web_search import search_web as _search_web

if TYPE_CHECKING:
    from models.schemas import GeopoliticalAnalysis, SectorAnalysis

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool(strict_mode=False)
def search_for_more_context(query: str) -> str:
    """Search the web for additional context to supplement the report when answering a question.

    Only call this if the full report context does not contain enough information
    to answer the question well.

    Args:
        query: Specific search query for the follow-up topic

    Returns:
        Top search results concatenated as text with title and snippet.
    """
    results = _search_web(query, max_results=4)
    if not results:
        return "No results found."
    parts = []
    for r in results:
        parts.append(f"**{r.get('title', 'Untitled')}**\n{r.get('snippet', '')}")
    return "\n\n".join(parts)


@function_tool(strict_mode=False)
def consult_specialist(specialist_type: str, question: str) -> str:
    """Re-query a specialist agent for a deeper, targeted answer on a specific topic.

    Use this when the user's follow-up question requires domain expertise
    beyond what's in the report (e.g. very recent news, highly technical detail).

    Args:
        specialist_type: 'geopolitical' for policy/trade questions,
                         'sector' for technology/competitive questions
        question: The specific question to ask the specialist

    Returns:
        The specialist's answer as plain text.
    """
    specialist_type = specialist_type.lower().strip()

    async def _run_geo():
        from pipeline.specialists.geopolitical_advisor import build_geopolitical_advisor_agent
        from agents import Runner, ItemHelpers
        from agents.items import MessageOutputItem
        agent = build_geopolitical_advisor_agent()
        result = await Runner.run(agent, input=question)
        raw = result.final_output
        if raw and str(raw).strip():
            return str(raw)
        for item in reversed(result.new_items):
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text.strip():
                    return text
        return "No response from specialist."

    async def _run_sector(sector: str):
        from pipeline.specialists.sector_specialist import build_sector_specialist_agent
        from agents import Runner, ItemHelpers
        from agents.items import MessageOutputItem
        agent = build_sector_specialist_agent(sector)
        result = await Runner.run(agent, input=question)
        raw = result.final_output
        if raw and str(raw).strip():
            return str(raw)
        for item in reversed(result.new_items):
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text.strip():
                    return text
        return "No response from specialist."

    loop = asyncio.new_event_loop()
    try:
        if specialist_type == "geopolitical":
            return loop.run_until_complete(_run_geo())
        else:
            # sector specialist — use a general fallback if no sector context available
            return loop.run_until_complete(_run_sector(_current_sector or "general"))
    except Exception as exc:
        return f"Specialist consultation failed: {exc}"
    finally:
        loop.close()


# ── Module-level sector state (set by build_qa_agent) ────────────────────────
_current_sector: str = ""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_qa_agent(
    full_report: str,
    sector: str,
    tickers: list[str],
    geo_analysis: "GeopoliticalAnalysis | None" = None,
    sector_analysis: "SectorAnalysis | None" = None,
) -> Agent:
    """Build the Q&A agent with the full report context injected.

    The report text and specialist analyses are embedded in the agent's instructions
    so the model can answer questions without needing tool calls for basic lookups.

    Args:
        full_report: Full markdown text of the generated analyst report
        sector: Sector name (used for specialist routing)
        tickers: List of tickers analyzed (for context)
        geo_analysis: Geopolitical analysis from the pipeline run (optional)
        sector_analysis: Sector specialist analysis from the pipeline run (optional)
    """
    global _current_sector
    _current_sector = sector

    # Build specialist context sections
    geo_section = ""
    if geo_analysis is not None:
        try:
            geo_section = f"\n\n=== GEOPOLITICAL ADVISOR ANALYSIS ===\n{geo_analysis.model_dump_json(indent=2)}"
        except Exception:
            pass

    specialist_section = ""
    if sector_analysis is not None:
        try:
            specialist_section = f"\n\n=== SECTOR SPECIALIST ANALYSIS ===\n{sector_analysis.model_dump_json(indent=2)}"
        except Exception:
            pass

    ticker_list = ", ".join(tickers[:15]) if tickers else "unknown"

    instructions = f"""\
You are the Q&A Agent for the Sector Analyst system.

You have access to a full analyst report and specialist analyses for the {sector} sector
(companies analyzed: {ticker_list}).

━━ FULL ANALYST REPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{full_report if full_report else "No report available — answer based on specialist analyses below."}

{geo_section}
{specialist_section}

━━ YOUR JOB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You operate in two modes based on the user's input:

MODE 1 — SUMMARIZE (triggered when user says "summarize"):
  Produce a concise 3-page executive summary containing:
  1. Sector Snapshot — key stats table (recommendation, sector size, key companies, forward P/E)
  2. Top 3 Key Findings — the most important data-backed conclusions with numbers
  3. Investment Thesis — one paragraph with the explicit Overweight/Neutral/Underweight recommendation
  4. Key Risks — top 3 risks in bullet form
  5. Geopolitical Context — one paragraph on the most important policy factors
  The summary should be self-contained and readable without the full report.

MODE 2 — ANSWER (triggered for any other question):
  - Answer directly from the report context above
  - Cite which section of the report your answer comes from
  - If the report doesn't contain enough detail, call search_for_more_context()
  - For very technical domain questions, call consult_specialist('geopolitical' or 'sector', question)
  - Be specific: include numbers, company names, dates from the report
  - Do NOT make up information not in the report or search results

━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After any tool calls, output ONLY this JSON (no markdown fences):

{{
  "question": "<the user's question or 'summarize'>",
  "answer": "<your full answer or executive summary in markdown format>",
  "answer_type": "<'summary' if summarize command, 'answer' otherwise>",
  "artifact_paths": [],
  "sources_consulted": ["<section name or URL if you searched>"]
}}
"""

    return Agent(
        name="QA",
        model=LitellmModel(model=LITELLM_MODEL_ID),
        instructions=instructions,
        tools=[search_for_more_context, consult_specialist],
    )
