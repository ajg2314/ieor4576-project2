"""Orchestrator — coordinates the four-agent pipeline.

Pipeline: Planner → Collector → EDA → Hypothesis

The Planner expands the user's question and identifies 10–20 sector companies.
The Collector fetches all their SEC EDGAR financial data.
The EDA agent analyses a compact annual-only view of the data and makes charts.
The Hypothesis agent synthesises a final analyst report.

Yields SSE-style (event_type, payload) tuples for live frontend streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, AsyncGenerator, TypeVar, Type

from agents import Runner, RunResult, ItemHelpers
from agents.items import MessageOutputItem
from pydantic import BaseModel

try:
    from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
except ImportError:
    LiteLLMRateLimitError = None  # type: ignore

import litellm

from .planner import build_planner_agent
from .collector import build_collector_agent
from .eda_agent import build_eda_agent, clear_eda_store, get_eda_observations, load_eda_data
from .hypothesis_agent import build_hypothesis_agent
from models.schemas import SectorPlan, DataBundle, EDAFindings, EDAFinding, HypothesisReport
from tools.sec_edgar import clear_record_store, get_stored_records

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
MAX_REFINEMENT_LOOPS = 2
RATE_LIMIT_MAX_WAIT = 60      # cap per-retry wait at 60 seconds
RATE_LIMIT_INITIAL_WAIT = 15  # first wait before retry
INTER_STEP_DELAY = 3          # seconds between pipeline steps to reduce burst

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

def _sanitise_json(text: str) -> str:
    """Fix common LLM JSON errors before parsing.

    - Strip markdown code fences
    - Remove invalid escape sequences (e.g. \' which is not valid JSON)
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    # \' is not a valid JSON escape — apostrophes need no escaping in JSON
    text = text.replace("\\'", "'")
    # Remove other invalid single-char escapes that are not in the JSON spec
    # Valid: \" \\ \/ \b \f \n \r \t \uXXXX
    text = re.sub(r"\\([^\"\\\/bfnrtu])", r"\1", text)
    return text


def _parse_text(raw: str, model_class: Type[T]) -> T:
    """Parse a raw string into a Pydantic model (strips fences, regex fallback)."""
    if not raw.strip():
        raise ValueError(
            f"Agent produced no text output for {model_class.__name__}."
        )
    cleaned = _sanitise_json(raw)
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

def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception is a 429 / resource-exhausted error."""
    if LiteLLMRateLimitError and isinstance(exc, LiteLLMRateLimitError):
        return True
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "ratelimit" in msg
        or "resource_exhausted" in msg
        or "resource exhausted" in msg
        or "too many requests" in msg
    )


async def _run_agent(
    agent,
    input: Any,
    model_class: Type[T],
    *,
    _status_sink: list[str] | None = None,
) -> T:
    """Run an agent with unlimited 429 retries and optional nudge for empty output.

    Rate-limit waits are appended to _status_sink (a list) so the caller can
    yield them as frontend status messages between steps.

    Backs off with capped exponential + jitter: 15s, 30s, 45s, 60s, 60s, ...
    Never gives up on rate limits — only raises on real errors.
    """
    attempt = 0
    while True:
        try:
            result = await Runner.run(agent, input=input)
        except Exception as exc:
            if _is_rate_limit(exc):
                wait = min(
                    RATE_LIMIT_INITIAL_WAIT * (2 ** min(attempt, 4))
                    + random.uniform(0, 5),
                    RATE_LIMIT_MAX_WAIT,
                )
                wait_int = int(wait)
                msg = (
                    f"Rate limit from Vertex AI — waiting {wait_int}s before retry "
                    f"(attempt {attempt + 1})..."
                )
                logger.warning("Rate limit on %s (attempt %d) — waiting %ds", agent.name, attempt + 1, wait_int)
                if _status_sink is not None:
                    _status_sink.append(msg)
                await asyncio.sleep(wait)
                attempt += 1
                continue
            raise  # non-rate-limit errors propagate immediately

        raw = _extract_text(result)

        # Nudge if model stopped after tool calls without emitting text
        for nudge_num in range(2):
            if raw.strip():
                break
            logger.warning("%s: no text output (nudge %d)", agent.name, nudge_num + 1)
            template = _json_template(model_class)
            nudge = (
                "Your tool calls are complete. Now write your final response.\n"
                "Output ONLY valid JSON matching this structure (fill in real values):\n"
                f"{template}\n"
                "No markdown fences. No other text. Just the JSON."
            )
            continued = result.to_input_list() + [{"role": "user", "content": nudge}]
            try:
                result = await Runner.run(agent, input=continued)
            except Exception as exc:
                if _is_rate_limit(exc):
                    wait = min(RATE_LIMIT_INITIAL_WAIT + random.uniform(0, 5), RATE_LIMIT_MAX_WAIT)
                    wait_int = int(wait)
                    if _status_sink is not None:
                        _status_sink.append(f"Rate limit during nudge — waiting {wait_int}s...")
                    await asyncio.sleep(wait)
                    attempt += 1
                    break  # retry outer loop
                raise
            raw = _extract_text(result)

        if not raw.strip():
            # Outer loop will retry from scratch
            attempt += 1
            continue

        return _parse_text(raw, model_class)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _compact_for_eda(bundle: DataBundle) -> DataBundle:
    """Filter DataBundle to deduplicated annual 10-K records for the last EDA_MAX_YEARS years.

    Deduplication key: (ticker, fiscal_year, metric) — one clean row per data point.
    This is the final safety net; get_stored_records() already deduplicates on
    (ticker, period, metric), but different period end-dates can share a fiscal_year.
    """
    annual = [r for r in bundle.records if r.get("form") == "10-K"]
    if not annual:
        annual = bundle.records  # fallback: keep all if no annual data

    years = sorted(set(r.get("fiscal_year", "0000") for r in annual), reverse=True)
    keep_years = set(years[:EDA_MAX_YEARS])
    filtered = [r for r in annual if r.get("fiscal_year") in keep_years]

    # Deduplicate: one record per (ticker, fiscal_year, metric), keep latest period
    seen: dict[tuple, dict] = {}
    for r in sorted(filtered, key=lambda x: x.get("period", "")):
        key = (r.get("ticker"), r.get("fiscal_year"), r.get("metric"))
        seen[key] = r  # later period (more recent filing) wins
    deduped = list(seen.values())

    logger.info(
        "EDA compact: %d total → %d annual → %d year-filtered → %d deduplicated (%s)",
        len(bundle.records), len(annual), len(filtered), len(deduped),
        ", ".join(sorted(keep_years, reverse=True)),
    )
    return DataBundle(
        source=bundle.source,
        retrieval_method=bundle.retrieval_method,
        records=deduped,
        metadata=bundle.metadata,
        summary=bundle.summary,
    )


async def _synthesise_eda_findings(
    question: str,
    sector: str,
    observations: list[dict],
) -> EDAFindings:
    """Stage 2 of EDA: fresh LLM call to synthesise observations into EDAFindings JSON.

    This call has a small, clean context (~1-2KB) so it reliably produces output,
    even when Stage 1 (the tool-calling agent) produced nothing.
    """
    if not observations:
        return EDAFindings(
            findings=[],
            key_insight="No tool results captured — data may be insufficient.",
            recommended_hypothesis_direction="Collect more data and retry.",
        )

    obs_text = "\n".join(
        f"- [{o['tool']}] {o['description']}: {o['value']}"
        + (f" → saved to {o['artifact_path']}" if o.get("artifact_path") else "")
        for o in observations
    )

    chart_obs = [o for o in observations if o.get("artifact_path")]

    synthesis_prompt = f"""\
You are summarising the results of an exploratory data analysis of {sector} companies.

User question: {question}

Tool call results from the EDA phase:
{obs_text}

Based ONLY on the above results, produce a JSON EDAFindings object.
Include one finding entry per tool call above. For chart tool calls, set artifact_path to the path shown.

Output ONLY this JSON, no other text:
{{
  "findings": [
    {{"tool_name": "create_chart", "description": "Revenue Trend", "value": "see chart", "artifact_path": "artifacts/revenue_trend_xxx.png"}},
    {{"tool_name": "run_python", "description": "...", "value": "...", "artifact_path": null}}
  ],
  "key_insight": "<single most important pattern with specific numbers from the results above>",
  "recommended_hypothesis_direction": "<what the hypothesis agent should focus on>"
}}"""

    model_id = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
    attempt = 0
    while True:
        try:
            resp = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or ""
            findings = _parse_text(raw, EDAFindings)
            # Ensure chart artifact paths are present even if LLM missed them
            chart_paths_in_findings = {f.artifact_path for f in findings.findings if f.artifact_path}
            for obs in chart_obs:
                if obs["artifact_path"] not in chart_paths_in_findings:
                    findings.findings.append(EDAFinding(
                        tool_name="create_chart",
                        description=obs["description"],
                        value=obs["artifact_path"],
                        artifact_path=obs["artifact_path"],
                    ))
            return findings
        except Exception as exc:
            if _is_rate_limit(exc):
                wait = min(
                    RATE_LIMIT_INITIAL_WAIT * (2 ** min(attempt, 4)) + random.uniform(0, 5),
                    RATE_LIMIT_MAX_WAIT,
                )
                logger.warning("Rate limit on EDA synthesis (attempt %d) — waiting %ds", attempt + 1, int(wait))
                await asyncio.sleep(wait)
                attempt += 1
            else:
                raise


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

    def _drain(sink: list[str]):
        """Yield and clear any rate-limit status messages queued during an agent call."""
        msgs, sink[:] = list(sink), []
        return msgs

    rate_sink: list[str] = []  # rate-limit wait messages accumulate here

    # ── Step 0: Plan ─────────────────────────────────────────────────────
    yield "status", "Step 1/4 — Researching sector and identifying key companies..."

    plan = await _run_agent(planner, full_question, SectorPlan, _status_sink=rate_sink)
    for msg in _drain(rate_sink):
        yield "status", msg

    yield "status", (
        f"Research plan: {plan.sector} | "
        f"{len(plan.tickers)} companies identified: {', '.join(plan.tickers[:8])}"
        + (f" +{len(plan.tickers)-8} more" if len(plan.tickers) > 8 else "")
    )

    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Step 1: Collect ───────────────────────────────────────────────────
    yield "status", f"Step 2/4 — Fetching SEC EDGAR data for {len(plan.tickers)} companies..."

    collector_brief = (
        f"Research brief from Planner:\n"
        f"Sector: {plan.sector}\n"
        f"Question: {plan.expanded_query}\n"
        f"Tickers to fetch (ALL of them): {json.dumps(plan.tickers)}\n"
        f"Focus metrics: {json.dumps(plan.focus_metrics)}\n\n"
        f"Original user question: {full_question}"
    )

    clear_record_store()
    data_bundle = await _run_agent(collector, collector_brief, DataBundle, _status_sink=rate_sink)
    for msg in _drain(rate_sink):
        yield "status", msg

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
        logger.warning("Side-channel store empty — falling back to LLM output (%d rows)", len(data_bundle.records))

    yield "status", (
        f"Data collected: {len(data_bundle.records)} records across "
        f"{len(set(r.get('ticker','') for r in data_bundle.records))} companies"
    )

    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Steps 2+3 with optional refinement loop ───────────────────────────
    eda_findings: EDAFindings | None = None
    compact_bundle = _compact_for_eda(data_bundle)

    for loop in range(MAX_REFINEMENT_LOOPS):
        loop_label = f" (refinement #{loop})" if loop > 0 else ""
        yield "status", f"Step 3/4 — Running exploratory data analysis{loop_label}..."

        # ── Stage 1: tool calls ───────────────────────────────────────────
        # load_eda_data writes records to disk, pre-generates standard charts,
        # and seeds _eda_observations with those auto charts.
        data_summary = load_eda_data(compact_bundle.records)
        # clear_eda_store() is a no-op — store was reset by load_eda_data.
        eda_input = (
            f"User question: {question}\n\n"
            f"Sector: {plan.sector} | Companies: {', '.join(plan.tickers)}\n\n"
            f"{data_summary}\n"
            f"Call your tools now. Tools read the data automatically — "
            f"do not pass records as parameters, just pass metric names and tickers."
        )
        eda_findings = None
        try:
            # EDA Stage 1 doesn't go through _run_agent because we don't require
            # text output — charts/stats are captured in the EDA side-channel store.
            attempt = 0
            while True:
                try:
                    result = await Runner.run(eda_agent, input=eda_input)
                    break
                except Exception as exc:
                    if _is_rate_limit(exc):
                        wait = min(
                            RATE_LIMIT_INITIAL_WAIT * (2 ** min(attempt, 4)) + random.uniform(0, 5),
                            RATE_LIMIT_MAX_WAIT,
                        )
                        wait_int = int(wait)
                        msg = f"Rate limit — waiting {wait_int}s before EDA retry {attempt + 1}..."
                        logger.warning(msg)
                        yield "status", msg
                        await asyncio.sleep(wait)
                        attempt += 1
                    else:
                        raise

            raw = _extract_text(result)
            if raw.strip():
                try:
                    eda_findings = _parse_text(raw, EDAFindings)
                    for obs in get_eda_observations():
                        if obs.get("artifact_path"):
                            existing = {f.artifact_path for f in eda_findings.findings}
                            if obs["artifact_path"] not in existing:
                                eda_findings.findings.append(EDAFinding(
                                    tool_name="create_chart",
                                    description=obs["description"],
                                    value=obs["artifact_path"],
                                    artifact_path=obs["artifact_path"],
                                ))
                except Exception:
                    eda_findings = None
        except Exception as exc:
            logger.error("EDA Stage 1 failed: %s", exc)

        # ── Stage 2: synthesis ────────────────────────────────────────────
        if eda_findings is None:
            yield "status", "EDA tool calls complete — synthesising findings..."
            await asyncio.sleep(INTER_STEP_DELAY)
            observations = get_eda_observations()
            logger.info("EDA synthesis from %d observations", len(observations))
            eda_findings = await _synthesise_eda_findings(question, plan.sector, observations)

        yield "status", f"Key insight: {eda_findings.key_insight[:140]}..."

        await asyncio.sleep(INTER_STEP_DELAY)

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
            additional = await _run_agent(collector, refinement_prompt, DataBundle, _status_sink=rate_sink)
            for msg in _drain(rate_sink):
                yield "status", msg
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
    report = await _run_agent(hypothesis_agent, hypothesis_input, HypothesisReport, _status_sink=rate_sink)
    for msg in _drain(rate_sink):
        yield "status", msg

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
        "artifact_paths": [p.replace("artifacts/", "/files/") for p in chart_paths],
        "confidence": report.confidence,
        "title": report.title,
        "sector": plan.sector,
        "companies": plan.tickers,
    }
