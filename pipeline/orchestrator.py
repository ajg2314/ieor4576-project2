"""Orchestrator — coordinates the three-agent pipeline.

Multi-agent pattern: sequential pipeline coordinator.
Runs three specialized agents in order, piping structured outputs between them.
Yields SSE-style status events so the frontend can show live progress.
Can loop (Collect → EDA → Collect) up to MAX_REFINEMENT_LOOPS times.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, TypeVar, Type

from agents import Runner
from pydantic import BaseModel

from .collector import build_collector_agent
from .eda_agent import build_eda_agent
from .hypothesis_agent import build_hypothesis_agent
from models.schemas import DataBundle, EDAFindings, HypothesisReport

T = TypeVar("T", bound=BaseModel)
MAX_REFINEMENT_LOOPS = 2


def _parse_agent_output(raw: str | object, model_class: Type[T]) -> T:
    """Parse an agent's string output into a Pydantic model.

    Tries in order:
    1. Already the right type (pass through)
    2. Direct JSON parse
    3. Extract first JSON object from mixed text (with markdown fence stripping)
    """
    if isinstance(raw, model_class):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return model_class.model_validate_json(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return model_class.model_validate_json(match.group())
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

    collect_result = await Runner.run(collector, input=full_question)
    data_bundle = _parse_agent_output(collect_result.final_output, DataBundle)

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
        eda_result = await Runner.run(eda_agent, input=eda_input)
        eda_findings = _parse_agent_output(eda_result.final_output, EDAFindings)

        yield "status", f"Key insight: {eda_findings.key_insight[:120]}..."

        if _needs_refinement(eda_findings) and loop < MAX_REFINEMENT_LOOPS - 1:
            yield "status", "EDA found data gaps — collecting additional data..."
            refinement_prompt = (
                f"{question}\n\n"
                f"Previous collection summary: {data_bundle.summary}\n\n"
                f"EDA found gaps: {eda_findings.recommended_hypothesis_direction}\n"
                "Please collect additional data to fill these gaps."
            )
            collect_result = await Runner.run(collector, input=refinement_prompt)
            additional = _parse_agent_output(collect_result.final_output, DataBundle)
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
    hypothesis_result = await Runner.run(hypothesis_agent, input=hypothesis_input)
    report = _parse_agent_output(hypothesis_result.final_output, HypothesisReport)

    yield "status", "Done — report ready."
    yield "result", {
        "hypothesis": report.hypothesis,
        "evidence": [e.model_dump() for e in report.evidence],
        "narrative": report.narrative,
        "artifact_paths": [p.replace("artifacts/", "/files/") for p in report.artifact_paths],
        "confidence": report.confidence,
        "title": report.title,
    }
