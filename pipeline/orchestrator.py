"""Orchestrator — coordinates the four-agent pipeline.

Pipeline: Planner → Collector → EDA → Hypothesis

The Planner expands the user's question and identifies 10–20 sector companies.
The Collector fetches all their SEC EDGAR financial data.
The EDA agent analyses a compact annual-only view of the data and makes charts.
The Hypothesis agent synthesises a final analyst report.

Yields SSE-style (event_type, payload) tuples for live frontend streaming.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, TypeVar, Type

from agents import Runner, RunResult, ItemHelpers
from agents.items import MessageOutputItem
from pydantic import BaseModel

from .planner import build_planner_agent
from .collector import build_collector_agent
from .eda_agent import build_eda_agent
from .hypothesis_agent import build_hypothesis_agent
from models.schemas import SectorPlan, DataBundle, EDAFindings, HypothesisReport
from tools.sec_edgar import clear_record_store, get_stored_records

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
MAX_REFINEMENT_LOOPS = 2

# Maximum annual fiscal years to pass to the EDA agent (keeps context manageable)
EDA_MAX_YEARS = 7


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(result: RunResult) -> str:
    """Extract the last non-empty assistant text from a RunResult."""
    raw = result.final_output
    if raw and str(raw).strip():
        return str(raw)
    for item in reversed(result.new_items):
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text.strip():
                return text
    return ""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_text(raw: str, model_class: Type[T]) -> T:
    """Parse a raw string into a Pydantic model (strips fences, regex fallback)."""
    if not raw.strip():
        raise ValueError(
            f"Agent produced no text output for {model_class.__name__}."
        )
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return model_class.model_validate_json(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return model_class.model_validate_json(match.group())
            except Exception:
                pass
        raise ValueError(
            f"Could not parse agent output as {model_class.__name__}.\n"
            f"Raw output (first 800 chars):\n{raw[:800]}"
        )


def _json_template(model_class: Type[T]) -> str:
    """Return a minimal JSON template string for the agent to fill."""
    templates = {
        SectorPlan: (
            '{\n  "sector": "...",\n  "expanded_query": "...",\n'
            '  "tickers": ["TICK1", "TICK2", "..."],\n'
            '  "rationale": "...",\n  "focus_metrics": ["revenue", "gross_profit", "operating_income"]\n}'
        ),
        DataBundle: (
            '{\n  "source": "SEC EDGAR XBRL API",\n  "retrieval_method": "api",\n'
            '  "records": [],\n'
            '  "metadata": {"companies": [], "concepts": [], "mda_summary": ""},\n'
            '  "summary": "..."\n}'
        ),
        EDAFindings: (
            '{\n  "findings": [\n'
            '    {"tool_name": "create_chart", "description": "...", "value": "...", "artifact_path": "artifacts/..."}\n'
            '  ],\n'
            '  "key_insight": "...",\n'
            '  "recommended_hypothesis_direction": "..."\n}'
        ),
        HypothesisReport: (
            '{\n  "title": "...",\n  "hypothesis": "...",\n'
            '  "evidence": [{"claim": "...", "data_point": "...", "source": "SEC EDGAR 10-K"}],\n'
            '  "narrative": "...",\n  "artifact_paths": [],\n  "confidence": "high"\n}'
        ),
    }
    return templates.get(model_class, "{}")


# ── Agent runner ──────────────────────────────────────────────────────────────

async def _run_agent(agent, input, model_class: Type[T]) -> T:
    """Run an agent and parse its output.

    Gemini sometimes stops after tool calls without emitting a final text message.
    We send up to two nudges with an explicit JSON template before giving up.
    """
    result = await Runner.run(agent, input=input)
    raw = _extract_text(result)

    for attempt in range(2):
        if raw.strip():
            break
        logger.warning(
            "%s produced no text (attempt %d) — nudging for JSON output",
            agent.name, attempt + 1,
        )
        template = _json_template(model_class)
        nudge = (
            f"Your tool calls are complete. Now write your final response.\n"
            f"Output ONLY valid JSON matching this structure (fill in real values):\n"
            f"{template}\n"
            f"No markdown fences. No other text. Just the JSON."
        )
        continued = result.to_input_list() + [{"role": "user", "content": nudge}]
        result = await Runner.run(agent, input=continued)
        raw = _extract_text(result)

    if not raw.strip():
        raise ValueError(
            f"Agent '{agent.name}' produced no text output for {model_class.__name__} "
            f"after two nudges. Check the agent's prompt and model configuration."
        )

    return _parse_text(raw, model_class)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _compact_for_eda(bundle: DataBundle) -> DataBundle:
    """Filter DataBundle to annual 10-K records for the last EDA_MAX_YEARS fiscal years.

    The EDA agent receives this as its context. Keeping it to ~100–150 rows prevents
    context overflow that causes Gemini to stop without producing output.
    """
    annual = [r for r in bundle.records if r.get("form") == "10-K"]
    if not annual:
        annual = bundle.records  # fallback: keep all if no annual data

    years = sorted(set(r.get("fiscal_year", "0000") for r in annual), reverse=True)
    keep_years = set(years[:EDA_MAX_YEARS])
    filtered = [r for r in annual if r.get("fiscal_year") in keep_years]

    logger.info(
        "EDA compact: %d total → %d annual → %d after year filter (%s)",
        len(bundle.records), len(annual), len(filtered),
        ", ".join(sorted(keep_years, reverse=True)),
    )
    return DataBundle(
        source=bundle.source,
        retrieval_method=bundle.retrieval_method,
        records=filtered,
        metadata=bundle.metadata,
        summary=bundle.summary,
    )


def _needs_refinement(findings: EDAFindings) -> bool:
    gap_keywords = ["missing", "insufficient", "need more", "additional data", "no data"]
    return any(kw in findings.recommended_hypothesis_direction.lower() for kw in gap_keywords)


# ── Main pipeline ─────────────────────────────────────────────────────────────

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
    planner = build_planner_agent()
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

    # ── Step 0: Plan ─────────────────────────────────────────────────────
    yield "status", "Step 1/4 — Researching sector and identifying key companies..."

    plan = await _run_agent(planner, full_question, SectorPlan)

    yield "status", (
        f"Research plan: {plan.sector} | "
        f"{len(plan.tickers)} companies identified: {', '.join(plan.tickers[:8])}"
        + (f" +{len(plan.tickers)-8} more" if len(plan.tickers) > 8 else "")
    )

    # ── Step 1: Collect ───────────────────────────────────────────────────
    yield "status", f"Step 2/4 — Fetching SEC EDGAR data for {len(plan.tickers)} companies..."

    # Build a collector brief that includes the planner's researched company list
    collector_brief = (
        f"Research brief from Planner:\n"
        f"Sector: {plan.sector}\n"
        f"Question: {plan.expanded_query}\n"
        f"Tickers to fetch (ALL of them): {json.dumps(plan.tickers)}\n"
        f"Focus metrics: {json.dumps(plan.focus_metrics)}\n\n"
        f"Original user question: {full_question}"
    )

    clear_record_store()
    data_bundle = await _run_agent(collector, collector_brief, DataBundle)

    # Inject records from the side-channel (LLM output has records: [] to avoid truncation)
    stored = get_stored_records()
    if stored:
        data_bundle = DataBundle(
            source=data_bundle.source,
            retrieval_method=data_bundle.retrieval_method,
            records=stored,
            metadata=data_bundle.metadata,
            summary=data_bundle.summary,
        )
        logger.info("Injected %d records from side-channel store", len(stored))
    else:
        logger.warning(
            "Side-channel store empty — falling back to LLM output (%d rows)",
            len(data_bundle.records),
        )

    yield "status", (
        f"Data collected: {len(data_bundle.records)} records across "
        f"{len(set(r.get('ticker','') for r in data_bundle.records))} companies"
    )

    # ── Steps 2+3 with optional refinement loop ───────────────────────────
    eda_findings: EDAFindings | None = None
    compact_bundle = _compact_for_eda(data_bundle)

    for loop in range(MAX_REFINEMENT_LOOPS):
        loop_label = f" (refinement #{loop})" if loop > 0 else ""
        yield "status", f"Step 3/4 — Running exploratory data analysis{loop_label}..."

        eda_input = (
            f"User question: {question}\n\n"
            f"Sector plan: {plan.sector} | Companies: {', '.join(plan.tickers)}\n\n"
            f"Annual financial data ({len(compact_bundle.records)} records, "
            f"last {EDA_MAX_YEARS} fiscal years, 10-K only):\n"
            f"{compact_bundle.model_dump_json(indent=2)}"
        )
        eda_findings = await _run_agent(eda_agent, eda_input, EDAFindings)

        yield "status", f"Key insight: {eda_findings.key_insight[:140]}..."

        if _needs_refinement(eda_findings) and loop < MAX_REFINEMENT_LOOPS - 1:
            yield "status", "EDA found data gaps — collecting additional data..."
            refinement_prompt = (
                f"Research brief from Planner:\n"
                f"Sector: {plan.sector}\n"
                f"Question: {plan.expanded_query}\n"
                f"Tickers to fetch: {json.dumps(plan.tickers)}\n"
                f"Focus metrics: {json.dumps(plan.focus_metrics)}\n\n"
                f"Previous collection summary: {data_bundle.summary}\n\n"
                f"EDA found gaps: {eda_findings.recommended_hypothesis_direction}\n"
                "Please collect additional data to fill these gaps."
            )
            clear_record_store()
            additional = await _run_agent(collector, refinement_prompt, DataBundle)
            additional_records = get_stored_records()
            if additional_records:
                additional = DataBundle(
                    source=additional.source,
                    retrieval_method=additional.retrieval_method,
                    records=additional_records,
                    metadata=additional.metadata,
                    summary=additional.summary,
                )
            data_bundle = DataBundle(
                source=f"{data_bundle.source} + {additional.source}",
                retrieval_method=data_bundle.retrieval_method,
                records=data_bundle.records + additional.records,
                metadata={**data_bundle.metadata, **additional.metadata},
                summary=f"{data_bundle.summary}; {additional.summary}",
            )
            compact_bundle = _compact_for_eda(data_bundle)
            continue

        break

    # ── Step 3: Hypothesize ───────────────────────────────────────────────
    yield "status", "Step 4/4 — Generating analyst hypothesis report..."

    hypothesis_input = (
        f"User question: {question}\n\n"
        f"Sector: {plan.sector} | Companies analysed: {', '.join(plan.tickers)}\n\n"
        f"EDA findings:\n{eda_findings.model_dump_json(indent=2)}\n\n"
        f"Data summary: {data_bundle.summary}"
    )
    report = await _run_agent(hypothesis_agent, hypothesis_input, HypothesisReport)

    # Collect all chart paths from EDA findings + hypothesis report
    chart_paths: list[str] = []
    for finding in eda_findings.findings:
        if finding.artifact_path:
            chart_paths.append(finding.artifact_path)
    chart_paths.extend(report.artifact_paths)

    yield "status", "Done — report ready."
    yield "result", {
        "hypothesis": report.hypothesis,
        "evidence": [e.model_dump() for e in report.evidence],
        "narrative": report.narrative,
        "artifact_paths": [
            p.replace("artifacts/", "/files/") for p in chart_paths
        ],
        "confidence": report.confidence,
        "title": report.title,
        "sector": plan.sector,
        "companies": plan.tickers,
    }
