"""Orchestrator — coordinates the three-agent pipeline.

Multi-agent pattern: sequential pipeline coordinator.
Runs three specialized agents in order, piping structured outputs between them.
Yields SSE-style status events so the frontend can show live progress.
Can loop (Collect → EDA → Collect) up to MAX_REFINEMENT_LOOPS times.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncGenerator, TypeVar, Type

from agents import Runner, RunResult, ItemHelpers
from agents.items import MessageOutputItem
from pydantic import BaseModel

from .collector import build_collector_agent
from .eda_agent import build_eda_agent
from .hypothesis_agent import build_hypothesis_agent
from models.schemas import DataBundle, EDAFindings, HypothesisReport

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
MAX_REFINEMENT_LOOPS = 2


def _extract_text(result: RunResult) -> str:
    """Extract the last non-empty assistant text from a RunResult.

    Gemini via LiteLLM sometimes produces an empty final_output after a long
    tool-call chain. Falls back to scanning new_items via SDK helpers.
    """
    # 1. Try final_output first
    raw = result.final_output
    if raw and str(raw).strip():
        return str(raw)

    # 2. Collect all MessageOutputItem text in reverse, return last non-empty
    for item in reversed(result.new_items):
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text.strip():
                return text

    logger.warning("No non-empty assistant text found in RunResult.new_items")
    return ""


def _parse_agent_output(result: RunResult, model_class: Type[T]) -> T:
    """Extract and parse a RunResult into a Pydantic model.

    Tries in order:
    1. final_output is already the right type
    2. Extract text from message history, then direct JSON parse
    3. Regex: find first {...} block in mixed text
    """
    raw = _extract_text(result)

    if not raw.strip():
        raise ValueError(
            f"Agent produced no text output for {model_class.__name__}. "
            "The model may have stopped after tool calls without emitting a final message."
        )

    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        return model_class.model_validate_json(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return model_class.model_validate_json(match.group())
            except Exception:
                pass
        raise ValueError(
            f"Could not parse agent output as {model_class.__name__}.\n"
            f"Raw output (first 800 chars):\n{raw[:800]}"
        )


async def _run_agent(agent, input, model_class: Type[T]) -> T:
    """Run an agent and parse its output, with an automatic nudge if output is empty.

    Gemini via LiteLLM sometimes stops after tool calls without emitting a final
    text message. When that happens we continue the conversation with an explicit
    request for the JSON output.
    """
    result = await Runner.run(agent, input=input)
    raw = _extract_text(result)

    if not raw.strip():
        logger.warning(
            "%s produced no text output — sending follow-up nudge for JSON",
            agent.name,
        )
        nudge = (
            f"You have finished your tool calls. Now output your final response.\n"
            f"Output ONLY the JSON object as specified in your instructions. "
            f"No other text, no markdown fences."
        )
        continued = result.to_input_list() + [{"role": "user", "content": nudge}]
        result = await Runner.run(agent, input=continued)
        raw = _extract_text(result)

    return _parse_text(raw, model_class)


def _parse_text(raw: str, model_class: Type[T]) -> T:
    """Parse a raw string into a Pydantic model."""
    if not raw.strip():
        raise ValueError(
            f"Agent produced no text output for {model_class.__name__} even after nudge."
        )
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return model_class.model_validate_json(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return model_class.model_validate_json(match.group())
            except Exception:
                pass
        raise ValueError(
            f"Could not parse agent output as {model_class.__name__}.\n"
            f"Raw output (first 800 chars):\n{raw[:800]}"
        )


def _needs_refinement(findings: EDAFindings) -> bool:
    gap_keywords = ["missing", "insufficient", "need more", "additional data", "no data"]
    return any(kw in findings.recommended_hypothesis_direction.lower() for kw in gap_keywords)


async def run_analysis_with_status(
    question: str,
    prior_context: str | None = None,
) -> AsyncGenerator[tuple[str, object], None]:
    """
    Run the full pipeline, yielding (event_type, payload) tuples as progress is made.

    Event types:
      "status"  — payload is a str message shown in the UI
      "result"  — payload is the final HypothesisReport as a dict
      "error"   — payload is an error message string
    """
    collector = build_collector_agent()
    eda_agent = build_eda_agent()
    hypothesis_agent = build_hypothesis_agent()

    # Prepend prior conversation context for follow-up questions
    full_question = question
    if prior_context:
        full_question = (
            f"Prior analysis context:\n{prior_context}\n\n"
            f"Follow-up question: {question}"
        )

    # ── Step 1: Collect ──────────────────────────────────────────────────
    yield "status", "Step 1/3 — Collecting data from SEC EDGAR..."

    data_bundle = await _run_agent(collector, full_question, DataBundle)

    yield "status", f"Data collected: {data_bundle.summary[:120]}..."

    # ── Steps 2+3 with optional refinement loop ───────────────────────────
    eda_findings: EDAFindings | None = None

    for loop in range(MAX_REFINEMENT_LOOPS):
        loop_label = f" (refinement #{loop})" if loop > 0 else ""
        yield "status", f"Step 2/3 — Running exploratory data analysis{loop_label}..."

        eda_input = (
            f"User question: {question}\n\n"
            f"Collected data:\n{data_bundle.model_dump_json(indent=2)}"
        )
        eda_findings = await _run_agent(eda_agent, eda_input, EDAFindings)

        yield "status", f"Key insight: {eda_findings.key_insight[:120]}..."

        if _needs_refinement(eda_findings) and loop < MAX_REFINEMENT_LOOPS - 1:
            yield "status", "EDA found data gaps — collecting additional data..."
            refinement_prompt = (
                f"{question}\n\n"
                f"Previous collection summary: {data_bundle.summary}\n\n"
                f"EDA found gaps: {eda_findings.recommended_hypothesis_direction}\n"
                "Please collect additional data to fill these gaps."
            )
            additional = await _run_agent(collector, refinement_prompt, DataBundle)
            data_bundle = DataBundle(
                source=f"{data_bundle.source} + {additional.source}",
                retrieval_method=data_bundle.retrieval_method,
                records=data_bundle.records + additional.records,
                metadata={**data_bundle.metadata, **additional.metadata},
                summary=f"{data_bundle.summary}; {additional.summary}",
            )
            continue

        break

    # ── Step 3: Hypothesize ───────────────────────────────────────────────
    yield "status", "Step 3/3 — Generating analyst hypothesis report..."

    hypothesis_input = (
        f"User question: {question}\n\n"
        f"EDA findings:\n{eda_findings.model_dump_json(indent=2)}\n\n"
        f"Raw data summary: {data_bundle.summary}"
    )
    report = await _run_agent(hypothesis_agent, hypothesis_input, HypothesisReport)

    yield "status", "Done — report ready."
    yield "result", {
        "hypothesis": report.hypothesis,
        "evidence": [e.model_dump() for e in report.evidence],
        "narrative": report.narrative,
        "artifact_paths": [p.replace("artifacts/", "/files/") for p in report.artifact_paths],
        "confidence": report.confidence,
        "title": report.title,
    }
