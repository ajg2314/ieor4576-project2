"""Orchestrator — coordinates the three-agent pipeline.

Multi-agent pattern: sequential pipeline coordinator.
The orchestrator runs three specialized agents in order, piping structured
outputs between them. It can loop (Collect → EDA → Collect) if the EDA agent
surfaces data gaps, up to a fixed retry limit.

Each agent has its own system prompt and responsibilities:
  Collector    → DataBundle
  EDA Agent    → EDAFindings
  Hypothesis   → HypothesisReport
"""

from __future__ import annotations

import json
import os
import re
from typing import TypeVar, Type

from agents import Runner
from pydantic import BaseModel

from .collector import build_collector_agent
from .eda_agent import build_eda_agent
from .hypothesis_agent import build_hypothesis_agent
from models.schemas import DataBundle, EDAFindings, HypothesisReport

T = TypeVar("T", bound=BaseModel)


def _parse_agent_output(raw: str | object, model_class: Type[T]) -> T:
    """Parse an agent's string output into a Pydantic model.

    Tries in order:
    1. Already the right type (pass through)
    2. Direct JSON parse
    3. Extract first JSON object from mixed text
    """
    if isinstance(raw, model_class):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return model_class.model_validate_json(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return model_class.model_validate_json(match.group())
        raise ValueError(f"Could not parse agent output as {model_class.__name__}:\n{raw[:500]}")

MAX_REFINEMENT_LOOPS = 2


async def run_analysis(question: str) -> HypothesisReport:
    """
    Run the full Collect → EDA → Hypothesize pipeline for a user question.

    The coordinator may loop back to the Collector if EDA findings indicate
    data gaps (iterative refinement), up to MAX_REFINEMENT_LOOPS times.
    """
    collector = build_collector_agent()
    eda_agent = build_eda_agent()
    hypothesis_agent = build_hypothesis_agent()

    # ── Step 1: Collect ──────────────────────────────────────────────────
    collect_result = await Runner.run(collector, input=question)
    data_bundle: DataBundle = _parse_agent_output(collect_result.final_output, DataBundle)

    # ── Steps 2+3 with optional refinement loop ───────────────────────────
    for loop in range(MAX_REFINEMENT_LOOPS):
        # Step 2: EDA
        eda_input = (
            f"User question: {question}\n\n"
            f"Collected data:\n{data_bundle.model_dump_json(indent=2)}"
        )
        eda_result = await Runner.run(eda_agent, input=eda_input)
        eda_findings: EDAFindings = _parse_agent_output(eda_result.final_output, EDAFindings)

        # Check if EDA recommends collecting more data (iterative refinement)
        needs_more_data = _needs_refinement(eda_findings)
        if needs_more_data and loop < MAX_REFINEMENT_LOOPS - 1:
            refinement_prompt = (
                f"{question}\n\n"
                f"Previous collection summary: {data_bundle.summary}\n\n"
                f"EDA found gaps: {eda_findings.recommended_hypothesis_direction}\n"
                "Please collect additional data to fill these gaps."
            )
            collect_result = await Runner.run(collector, input=refinement_prompt)
            additional: DataBundle = _parse_agent_output(collect_result.final_output, DataBundle)
            # Merge records from both collections
            data_bundle = DataBundle(
                source=f"{data_bundle.source} + {additional.source}",
                retrieval_method=data_bundle.retrieval_method,
                records=data_bundle.records + additional.records,
                metadata={**data_bundle.metadata, **additional.metadata},
                summary=f"{data_bundle.summary}; {additional.summary}",
            )
            continue  # re-run EDA with enriched data

        break  # EDA is sufficient, proceed to hypothesis

    # ── Step 3: Hypothesize ───────────────────────────────────────────────
    hypothesis_input = (
        f"User question: {question}\n\n"
        f"EDA findings:\n{eda_findings.model_dump_json(indent=2)}\n\n"
        f"Raw data summary: {data_bundle.summary}"
    )
    hypothesis_result = await Runner.run(hypothesis_agent, input=hypothesis_input)
    report: HypothesisReport = _parse_agent_output(hypothesis_result.final_output, HypothesisReport)

    return report


def _needs_refinement(findings: EDAFindings) -> bool:
    """Heuristic: does the EDA recommendation signal missing data?"""
    gap_keywords = ["missing", "insufficient", "need more", "additional data", "no data"]
    direction = findings.recommended_hypothesis_direction.lower()
    return any(kw in direction for kw in gap_keywords)
