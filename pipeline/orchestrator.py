"""Orchestrator — coordinates the multi-agent pipeline.

Pipeline:
  Planner → Peer Discovery → [PARALLEL: Researcher + Valuation + Sentiment]
  → Collector → EDA → Hypothesis

The Planner expands the user's question and identifies 10–20 sector companies.
Peer Discovery validates tickers via yfinance and sorts by market cap.
Researcher, Valuation, and Sentiment agents run in parallel for speed.
The Collector fetches all SEC EDGAR financial data.
The EDA agent analyses a compact annual-only view of the data and makes charts.
The Hypothesis agent synthesises a final analyst report with live valuation data.

Yields SSE-style (event_type, payload) tuples for live frontend streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import date
from pathlib import Path
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
from .researcher import build_researcher_agent
from .collector import build_collector_agent
from .eda_agent import (
    build_eda_agent, clear_eda_store, get_eda_observations,
    load_eda_db, load_eda_data, set_eda_db_path,
)
from .hypothesis_agent import build_hypothesis_agent, get_last_saved_report_path
import pipeline.hypothesis_agent as _ha_module
from .peer_discovery import discover_peers
from .valuation_agent import build_valuation_agent
from .sentiment_agent import build_sentiment_agent
from .specialists.geopolitical_advisor import build_geopolitical_advisor_agent
from .specialists.sector_specialist import build_sector_specialist_agent
from models.schemas import (
    SectorPlan, ResearchContext, DataBundle, EDAFindings, EDAFinding,
    HypothesisReport, ValuationContext, SentimentContext,
    GeopoliticalAnalysis, SectorAnalysis,
)
from tools.sec_edgar import clear_record_store, get_stored_records
from tools.rag_store import seed_all as _seed_rag
from tools.market_data import clear_market_data_observations

# ── Q&A context store (module-level, retrieved by qa_cli.py) ─────────────────
_last_run_context: dict = {}


def get_last_run_context() -> dict:
    """Return the context from the most recent pipeline run for Q&A agent use."""
    return dict(_last_run_context)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
MAX_REFINEMENT_LOOPS = 1
RATE_LIMIT_MAX_WAIT = 60      # cap per-retry wait at 60 seconds
RATE_LIMIT_INITIAL_WAIT = 15  # first wait before retry
INTER_STEP_DELAY = 3          # seconds between pipeline steps to reduce burst

# Maximum annual fiscal years to pass to the EDA agent (keeps context manageable)
EDA_MAX_YEARS = 4

# ── LiteLLM burst throttle ────────────────────────────────────────────────────
# Limits concurrent in-flight litellm.acompletion() calls across ALL agents.
# Vertex AI free tier: ~2 RPS burst → keep MAX_CONCURRENT_LLM_CALLS ≤ 4.
# Raise MAX_CONCURRENT_LLM_CALLS / lower MIN_REQUEST_INTERVAL on paid tiers.
MAX_CONCURRENT_LLM_CALLS = 2   # max simultaneous API calls in flight
MIN_REQUEST_INTERVAL    = 1.0   # minimum seconds between successive dispatches

# Lazily created inside the event loop (asyncio objects can't be created at import time)
_llm_semaphore: asyncio.Semaphore | None = None
_llm_dispatch_lock: asyncio.Lock | None = None
_llm_last_dispatch: float = 0.0


def _get_throttle() -> tuple[asyncio.Semaphore, asyncio.Lock]:
    """Return the shared semaphore and dispatch lock, creating them on first call."""
    global _llm_semaphore, _llm_dispatch_lock
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    if _llm_dispatch_lock is None:
        _llm_dispatch_lock = asyncio.Lock()
    return _llm_semaphore, _llm_dispatch_lock


def _apply_litellm_throttle() -> None:
    """Monkey-patch litellm.acompletion with a semaphore + minimum interval guard.

    Idempotent — checked via a sentinel attribute on the litellm module so the
    patch is applied at most once per process even if called on every pipeline run.

    Two-layer approach:
      asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS) — at most N calls in-flight.
      asyncio.Lock + timestamp                     — MIN_REQUEST_INTERVAL spacing
                                                     between successive dispatches.
    This combination prevents both the initial burst (all 5 parallel agents firing
    simultaneously) and mid-run bursts (multiple agents hitting LLM at the same time).

    The existing 429 exponential backoff in _run_agent() remains the second line
    of defence for any requests that still slip through.
    """
    if getattr(litellm, "_sector_analyst_throttle_applied", False):
        return  # already patched — idempotent

    _original_acompletion = litellm.acompletion

    async def _throttled_acompletion(*args, **kwargs):
        global _llm_last_dispatch
        sem, lock = _get_throttle()
        async with sem:
            # Enforce minimum spacing between successive dispatches.
            # The lock serializes the timestamp check; the sleep is inside the
            # lock so no two coroutines can both decide "I'm allowed now"
            # simultaneously.
            async with lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                gap = now - _llm_last_dispatch
                if gap < MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(MIN_REQUEST_INTERVAL - gap)
                _llm_last_dispatch = loop.time()
            # Call the real litellm.acompletion while holding the semaphore slot.
            # Exceptions propagate transparently — the 429 retry logic in
            # _run_agent() sees them exactly as before.
            return await _original_acompletion(*args, **kwargs)

    litellm.acompletion = _throttled_acompletion
    setattr(litellm, "_sector_analyst_throttle_applied", True)
    logger.info(
        "LiteLLM throttle applied: max_concurrent=%d  min_interval=%.2fs",
        MAX_CONCURRENT_LLM_CALLS, MIN_REQUEST_INTERVAL,
    )


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

    - Strip markdown code fences (handles ```json ... ``` blocks)
    - Remove invalid escape sequences (e.g. \' which is not valid JSON)
    """
    text = text.strip()
    # Strip leading ```json or ``` fence (may have whitespace before the brace)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Strip trailing ``` fence
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # \' is not a valid JSON escape — apostrophes need no escaping in JSON
    text = text.replace("\\'", "'")
    # Remove other invalid single-char escapes that are not in the JSON spec
    # Valid: \" \\ \/ \b \f \n \r \t \uXXXX
    text = re.sub(r"\\([^\"\\\/bfnrtu])", r"\1", text)
    return text


def _extract_json_object(text: str) -> str | None:
    """Find the last complete JSON object {...} in text (handles mixed markdown+JSON output)."""
    # Find all {...} blocks, return the last parseable one
    candidates = list(re.finditer(r"\{", text))
    for start_match in reversed(candidates):
        start = start_match.start()
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:]):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : start + i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break  # try the next candidate
    return None


def _parse_text(raw: str, model_class: Type[T]) -> T:
    """Parse a raw string into a Pydantic model (strips fences, JSON extraction fallback)."""
    if not raw.strip():
        raise ValueError(
            f"Agent produced no text output for {model_class.__name__}."
        )
    cleaned = _sanitise_json(raw)
    try:
        return model_class.model_validate_json(cleaned)
    except Exception:
        # Try extracting the last valid JSON object from the text
        # (handles cases where the model outputs report prose + JSON)
        extracted = _extract_json_object(cleaned) or _extract_json_object(raw)
        if extracted:
            try:
                return model_class.model_validate_json(extracted)
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
        ResearchContext: (
            '{\n  "sector": "...",\n  "technology_context": "...",\n'
            '  "market_context": "...",\n  "geopolitical_context": "...",\n'
            '  "expert_sentiment": "...",\n'
            '  "key_risks": ["..."],\n  "qualitative_insights": ["..."],\n'
            '  "sources_consulted": [{"title": "...", "url": "...", "snippet": "..."}]\n}'
        ),
        HypothesisReport: (
            '{\n  "title": "...",\n  "hypothesis": "...",\n'
            '  "evidence": [{"claim": "...", "data_point": "...", "source": "SEC EDGAR 10-K"}],\n'
            '  "narrative": "...",\n  "artifact_paths": [],\n  "confidence": "high"\n}'
        ),
        GeopoliticalAnalysis: (
            '{\n  "sector": "...",\n  "key_policies": ["..."],\n'
            '  "company_exposures": [{"ticker": "...", "exposure": "high", "mechanism": "..."}],\n'
            '  "geographic_risks": ["..."],\n  "policy_tailwinds": ["..."],\n'
            '  "tail_risks": ["..."],\n  "summary": "..."\n}'
        ),
        SectorAnalysis: (
            '{\n  "sector": "...",\n  "specialist_type": "tech",\n'
            '  "technology_sota": "...",\n  "competitive_dynamics": "...",\n'
            '  "forward_thesis": "...",\n  "key_disruptions": ["..."],\n'
            '  "summary": "..."\n}'
        ),
    }
    return templates.get(model_class, "{}")


# ── Specialist context formatters ─────────────────────────────────────────────

def _format_geo_analysis(geo: "GeopoliticalAnalysis") -> str:
    """Format GeopoliticalAnalysis as readable analyst prose rather than raw JSON.

    The Hypothesis agent interprets this as natural text — avoids JSON field names
    (e.g. 'key_policies:') bleeding directly into the report body.
    """
    lines = [geo.summary]
    if geo.key_policies:
        lines.append("\nKey policies and regulations:")
        lines.extend(f"  • {p}" for p in geo.key_policies)
    if geo.company_exposures:
        lines.append("\nCompany-level exposure:")
        for e in geo.company_exposures:
            lines.append(f"  • {e.get('ticker','?')} ({e.get('exposure','?')} exposure): {e.get('mechanism','')}")
    if geo.policy_tailwinds:
        lines.append("\nPolicy tailwinds:")
        lines.extend(f"  • {t}" for t in geo.policy_tailwinds)
    if geo.tail_risks:
        lines.append("\nTail risks:")
        lines.extend(f"  • {r}" for r in geo.tail_risks)
    if geo.geographic_risks:
        lines.append("\nGeographic concentration risks:")
        lines.extend(f"  • {r}" for r in geo.geographic_risks)
    return "\n".join(lines)


def _format_sector_analysis(sa: "SectorAnalysis") -> str:
    """Format SectorAnalysis as readable analyst prose rather than raw JSON."""
    lines = [
        sa.summary,
        f"\nTechnology state-of-the-art:\n{sa.technology_sota}",
        f"\nCompetitive dynamics:\n{sa.competitive_dynamics}",
        f"\nForward investment thesis:\n{sa.forward_thesis}",
    ]
    if sa.key_disruptions:
        lines.append("\nKey disruptions to watch:")
        lines.extend(f"  • {d}" for d in sa.key_disruptions)
    return "\n".join(lines)


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

        # Try to parse; if it fails, nudge the model to output JSON and retry once
        try:
            return _parse_text(raw, model_class)
        except (ValueError, Exception) as parse_exc:
            logger.warning(
                "%s: text output not parseable as %s — nudging for JSON (attempt %d)\n%s",
                agent.name, model_class.__name__, attempt + 1, str(parse_exc)[:200],
            )
            template = _json_template(model_class)
            nudge = (
                "Your previous response could not be parsed as the required JSON structure.\n"
                "Do NOT repeat the report text. Output ONLY this JSON (fill in real values):\n"
                f"{template}\n"
                "No markdown. No prose. Just the JSON object."
            )
            continued = result.to_input_list() + [{"role": "user", "content": nudge}]
            try:
                result = await Runner.run(agent, input=continued)
                raw = _extract_text(result)
                if raw.strip():
                    try:
                        return _parse_text(raw, model_class)
                    except Exception:
                        pass  # fall through to retry outer loop
            except Exception as nudge_exc:
                if _is_rate_limit(nudge_exc):
                    wait = min(RATE_LIMIT_INITIAL_WAIT + random.uniform(0, 5), RATE_LIMIT_MAX_WAIT)
                    wait_int = int(wait)
                    if _status_sink is not None:
                        _status_sink.append(f"Rate limit during JSON nudge — waiting {wait_int}s...")
                    await asyncio.sleep(wait)
                    attempt += 1
                    continue
                raise
            # Still not parseable — retry outer loop from scratch
            attempt += 1
            continue


# ── Data helpers ──────────────────────────────────────────────────────────────

def _compact_for_eda(bundle: DataBundle) -> DataBundle:
    """Filter DataBundle to deduplicated annual 10-K records for the last EDA_MAX_YEARS years.

    Deduplication key: (ticker, fiscal_year, metric) — one clean row per data point.
    This is the final safety net; get_stored_records() already deduplicates on
    (ticker, period, metric), but different period end-dates can share a fiscal_year.
    """
    annual = [r for r in bundle.records if r.get("form") in ("10-K", "yfinance-annual")]
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

    # Filter out raw error content from observations — tracebacks and stderr
    # output from failed code executions must not appear in the EDA synthesis
    # prompt, as they can bleed into the hypothesis agent's report output.
    _error_markers = ("traceback", "error:", "exception:", "syntaxerror", "nameerror",
                      "keyerror", "valueerror", "typeerror", "stderr")

    def _clean_obs_value(val: str) -> str:
        if any(m in val.lower() for m in _error_markers):
            return "(code execution failed — skipped)"
        return val

    obs_text = "\n".join(
        f"- [{o['tool']}] {o['description']}: {_clean_obs_value(str(o['value']))}"
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
      "status"   — payload is a str message shown in the UI
      "progress" — payload is a dict with step, pct, elapsed_seconds, estimated_remaining_seconds
      "result"   — payload is the final HypothesisReport as a dict
      "error"    — payload is an error message string
    """
    # Apply LiteLLM burst throttle (idempotent — no-op after first call)
    _apply_litellm_throttle()

    # ── Timing setup ─────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    # Cumulative % complete at the START of each step (1-indexed, step 8 = done)
    _STEP_PCT = [0, 4, 8, 46, 62, 84, 88, 100]

    def _progress(step: int, name: str) -> tuple[str, dict]:
        elapsed = int(time.perf_counter() - _t0)
        pct = _STEP_PCT[step - 1]
        if pct > 5 and elapsed > 0:
            est_total = int(elapsed / (pct / 100))
            remaining = max(0, est_total - elapsed)
        else:
            remaining = 210  # default ~3.5 min estimate at pipeline start
        return ("progress", {
            "step": step, "total_steps": 7, "step_name": name,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
            "pct": pct,
        })

    # Seed RAG store on startup (idempotent — only adds new chunks)
    try:
        rag_counts = _seed_rag()
        if any(v > 0 for v in rag_counts.values()):
            logger.info("RAG seeded: %s", rag_counts)
    except Exception as exc:
        logger.warning("RAG seed failed (non-fatal): %s", exc)

    planner = build_planner_agent()
    researcher = build_researcher_agent()
    eda_agent = build_eda_agent()

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
    yield _progress(1, "Planning sector scope")
    yield "status", "Step 1/7 — Planning sector scope and identifying key companies..."

    plan = await _run_agent(planner, full_question, SectorPlan, _status_sink=rate_sink)
    for msg in _drain(rate_sink):
        yield "status", msg

    yield "status", (
        f"Sector: {plan.sector} | "
        f"{len(plan.tickers)} companies identified: {', '.join(plan.tickers[:8])}"
        + (f" +{len(plan.tickers)-8} more" if len(plan.tickers) > 8 else "")
    )
    logger.info("STEP_TIMING step=1 name=Plan elapsed=%.1f", time.perf_counter() - _t0)

    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Step 0.5: Peer Discovery ──────────────────────────────────────────
    yield _progress(2, "Validating peer universe")
    yield "status", "Step 2/7 — Validating peer universe with live market data..."

    clear_market_data_observations()
    peer_list = None
    try:
        peer_list = await discover_peers(plan)
        if peer_list.tickers:
            plan = plan.model_copy(update={"tickers": peer_list.tickers})
        top_peers = ", ".join(
            f"{p.ticker}(${p.market_cap_b:.0f}B)" if p.market_cap_b else p.ticker
            for p in (peer_list.peers[:5] if peer_list.peers else [])
            if p.valid
        )
        yield "status", (
            f"Peer validation complete: {len(peer_list.tickers)} valid companies | "
            f"{peer_list.selection_rationale[:120]}"
        )
        if top_peers:
            yield "status", f"Top peers by market cap: {top_peers}"
    except Exception as exc:
        logger.warning("Peer discovery failed (non-fatal): %s", exc)
        yield "status", "Peer validation skipped — using Planner tickers"

    logger.info("STEP_TIMING step=2 name=PeerDiscovery elapsed=%.1f", time.perf_counter() - _t0)
    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Step 1: Parallel — Research + Valuation + Sentiment + Specialists ────
    yield _progress(3, "Parallel research & specialist analysis")
    yield "status", (
        "Step 3/7 — Parallel: qualitative research + live valuation + sentiment + "
        "geopolitical advisor + sector specialist..."
    )

    today = date.today().isoformat()
    research_input = (
        f"Analysis date: {today}\n"
        f"Sector: {plan.sector}\n"
        f"User question: {plan.expanded_query}\n"
        f"Key companies to consider: {', '.join(plan.tickers[:12])}\n\n"
        f"Research the qualitative context for this sector — technology trends, "
        f"analyst views, competitive dynamics, regulatory environment, and expert insights. "
        f"Focus on developments from 2025 and 2026. Include the current year in your search queries."
    )
    valuation_input = (
        f"Analysis date: {today}\n"
        f"Sector: {plan.sector}\n"
        f"Tickers to fetch valuation data for: {json.dumps(plan.tickers)}\n\n"
        f"Fetch live valuation multiples and YTD returns for all tickers. "
        f"Build a comp table and interpret which companies look expensive vs cheap."
    )
    sentiment_input = (
        f"Analysis date: {today}\n"
        f"Sector: {plan.sector}\n"
        f"Key companies: {', '.join(plan.tickers[:8])}\n\n"
        f"Research current market sentiment for this sector as of {today}. "
        f"Search for recent earnings reactions, analyst upgrades/downgrades, "
        f"and the dominant investor narrative. Focus on 2025 and 2026 Q1 developments."
    )
    geo_input = (
        f"Analysis date: {today}\n"
        f"Sector: {plan.sector}\n"
        f"Key companies: {', '.join(plan.tickers[:12])}\n\n"
        f"Produce a rigorous geopolitical risk analysis for this sector as of {today}. "
        f"Research export controls, industrial policy, sanctions, geographic concentration, "
        f"and company-level exposure. Name specific policies with dates and quantified impact. "
        f"Include '2025' or '2026' in your search queries for the most recent policy developments."
    )
    specialist_input = (
        f"Analysis date: {today}\n"
        f"Sector: {plan.sector}\n"
        f"Key companies: {', '.join(plan.tickers[:12])}\n\n"
        f"Produce a domain expert analysis of technology SOTA and competitive dynamics "
        f"for this sector as of {today}. Focus on forward-looking investment thesis — what the market "
        f"is pricing in for the next 3-5 years and which companies are positioned to win. "
        f"Include the current year in your search queries to surface the most recent data."
    )

    research_context: ResearchContext | None = None
    valuation_context: ValuationContext | None = None
    sentiment_context: SentimentContext | None = None
    geo_analysis: GeopoliticalAnalysis | None = None
    sector_analysis: SectorAnalysis | None = None

    parallel_results = await asyncio.gather(
        _run_agent(researcher, research_input, ResearchContext, _status_sink=rate_sink),
        _run_agent(build_valuation_agent(), valuation_input, ValuationContext, _status_sink=rate_sink),
        _run_agent(build_sentiment_agent(), sentiment_input, SentimentContext, _status_sink=rate_sink),
        _run_agent(build_geopolitical_advisor_agent(), geo_input, GeopoliticalAnalysis, _status_sink=rate_sink),
        _run_agent(build_sector_specialist_agent(plan.sector), specialist_input, SectorAnalysis, _status_sink=rate_sink),
        return_exceptions=True,
    )

    for msg in _drain(rate_sink):
        yield "status", msg

    if not isinstance(parallel_results[0], Exception):
        research_context = parallel_results[0]
        yield "status", (
            f"Research complete: {len(research_context.sources_consulted)} sources | "
            f"{len(research_context.key_risks)} risks identified"
        )
    else:
        logger.warning("Researcher failed (non-fatal): %s", parallel_results[0])
        yield "status", "Web research skipped"

    if not isinstance(parallel_results[1], Exception):
        valuation_context = parallel_results[1]
        n_metrics = len(valuation_context.metrics)
        median_pe = valuation_context.sector_median_pe
        yield "status", (
            f"Valuation complete: {n_metrics} companies | "
            f"sector median P/E: {f'{median_pe:.1f}x' if median_pe else 'N/A'}"
        )
    else:
        logger.warning("Valuation agent failed (non-fatal): %s", parallel_results[1])
        yield "status", "Live valuation data unavailable"

    if not isinstance(parallel_results[2], Exception):
        sentiment_context = parallel_results[2]
        yield "status", (
            f"Sentiment: {sentiment_context.overall_sentiment} "
            f"(score: {sentiment_context.sentiment_score:+.2f}) | "
            f"themes: {', '.join(sentiment_context.key_themes[:3])}"
        )
    else:
        logger.warning("Sentiment agent failed (non-fatal): %s", parallel_results[2])
        yield "status", "Sentiment analysis unavailable"

    if not isinstance(parallel_results[3], Exception):
        geo_analysis = parallel_results[3]
        n_policies = len(geo_analysis.key_policies)
        n_exposures = len(geo_analysis.company_exposures)
        yield "status", (
            f"Geopolitical analysis complete: {n_policies} key policies | "
            f"{n_exposures} company exposures mapped"
        )
    else:
        logger.warning("Geopolitical advisor failed (non-fatal): %s", parallel_results[3])
        yield "status", "Geopolitical analysis unavailable"

    if not isinstance(parallel_results[4], Exception):
        sector_analysis = parallel_results[4]
        yield "status", (
            f"Sector specialist ({sector_analysis.specialist_type}) complete: "
            f"{len(sector_analysis.key_disruptions)} disruptions identified"
        )
    else:
        logger.warning("Sector specialist failed (non-fatal): %s", parallel_results[4])
        yield "status", "Sector specialist analysis unavailable"

    logger.info("STEP_TIMING step=3 name=ParallelResearch elapsed=%.1f", time.perf_counter() - _t0)
    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Step 4: Collect (deterministic — no LLM) ──────────────────────────
    # Bypasses the Collector LLM agent. The collector's job is pure data fetching
    # (the Planner already chose the tickers), so we call the same SEC tools
    # directly. Saves ~8 LLM calls and ~50K tokens of growing context that
    # otherwise blow past Vertex AI's per-minute token quota.
    yield _progress(4, "Fetching SEC EDGAR financial data")
    yield "status", f"Step 4/7 — Fetching SEC EDGAR data for {len(plan.tickers)} companies..."

    from tools.sec_edgar import get_sector_financials, get_recent_filing_text, append_to_record_store
    from tools.market_data import get_company_financials_yf

    clear_record_store()
    concepts = plan.focus_metrics or [
        "revenue", "net_income", "operating_income", "gross_profit", "rd_expense",
    ]
    sector_result = await asyncio.to_thread(get_sector_financials, plan.tickers, concepts)

    # yfinance fallback for any ticker EDGAR returned 0 records for
    edgar_records = get_stored_records()
    found_tickers = {r.get("ticker") for r in edgar_records}
    missing = [t for t in plan.tickers if t not in found_tickers]
    for t in missing:
        try:
            yf_result = await asyncio.to_thread(get_company_financials_yf, t)
            yf_records = yf_result.get("flat_records", [])
            if yf_records:
                append_to_record_store(yf_records)
        except Exception as exc:
            logger.warning("yfinance fallback failed for %s: %s", t, exc)

    # MD&A text from the largest ticker (10-K + 10-Q for forward guidance)
    mda_summary = ""
    if plan.tickers:
        try:
            tk = plan.tickers[0]
            r10k = await asyncio.to_thread(get_recent_filing_text, tk, "10-K")
            r10q = await asyncio.to_thread(get_recent_filing_text, tk, "10-Q")
            mda_summary = (
                f"10-K MD&A ({tk}): {r10k.get('text','')[:3000]}\n\n"
                f"10-Q MD&A ({tk}): {r10q.get('text','')[:3000]}"
            )
        except Exception as exc:
            logger.warning("Filing text fetch failed: %s", exc)

    stored = get_stored_records()
    data_bundle = DataBundle(
        source="SEC EDGAR XBRL API",
        retrieval_method="api",
        records=stored,
        metadata={
            "companies": plan.tickers,
            "concepts": concepts,
            "mda_summary": mda_summary,
        },
        summary=f"Fetched {len(stored)} records for {len(plan.tickers)} companies via direct SEC API.",
    )

    yield "status", (
        f"Data collected: {len(data_bundle.records)} records across "
        f"{len(set(r.get('ticker','') for r in data_bundle.records))} companies"
    )
    logger.info("STEP_TIMING step=4 name=EDGARCollect elapsed=%.1f", time.perf_counter() - _t0)

    await asyncio.sleep(INTER_STEP_DELAY)

    # ── Steps 3+4 with optional refinement loop ───────────────────────────
    eda_findings: EDAFindings | None = None
    compact_bundle = _compact_for_eda(data_bundle)

    # Use a per-run isolated DB file to prevent contamination between concurrent runs.
    run_id = uuid.uuid4().hex[:8]
    eda_db_path = str(Path("artifacts") / f"eda_{run_id}.db")
    set_eda_db_path(eda_db_path)

    for loop in range(MAX_REFINEMENT_LOOPS):
        loop_label = f" (refinement #{loop})" if loop > 0 else ""
        yield _progress(5, "Exploratory data analysis")
        yield "status", f"Step 5/7 — Running exploratory data analysis{loop_label}..."

        # ── Stage 1: tool calls ───────────────────────────────────────────
        # Load ALL records (10-K + 10-Q, full history) into SQLite so the
        # agent can query any slice it needs via sql_query / run_python.
        load_eda_db(data_bundle.records)
        # load_eda_data pre-generates standard charts and builds the summary.
        data_summary = load_eda_data(compact_bundle.records)
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
            yield _progress(6, "Synthesising EDA findings")
            yield "status", "Step 6/7 — EDA tool calls complete — synthesising findings..."
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

    logger.info("STEP_TIMING step=5 name=EDA elapsed=%.1f", time.perf_counter() - _t0)

    # ── Step 4: Hypothesize ───────────────────────────────────────────────
    yield _progress(7, "Generating analyst report")
    yield "status", "Step 7/7 — Generating analyst hypothesis report..."

    _ha_module._last_saved_report_path = ""  # reset side-channel before each run
    hypothesis_agent = build_hypothesis_agent(sector=plan.sector)

    research_section = ""
    if research_context is not None:
        geo = research_context.geopolitical_context
        research_section = (
            f"\n\nQualitative Research Context (from web research):\n"
            f"Technology context: {research_context.technology_context}\n\n"
            f"Market context: {research_context.market_context}\n\n"
            + (f"Geopolitical context: {geo}\n\n" if geo else "")
            + f"Expert sentiment: {research_context.expert_sentiment}\n\n"
            f"Key risks identified:\n"
            + "\n".join(f"- {r}" for r in research_context.key_risks)
            + "\n\nForward-looking qualitative insights:\n"
            + "\n".join(f"- {i}" for i in research_context.qualitative_insights)
            + f"\n\nSources consulted: {len(research_context.sources_consulted)} web sources "
            f"(analyst reports, news, expert commentary)"
        )

    valuation_section = (
        valuation_context.model_dump_json(indent=2)
        if valuation_context else "Not available — use EDA data for valuation estimates."
    )
    sentiment_section = (
        sentiment_context.model_dump_json(indent=2)
        if sentiment_context else "Not available — omit sentiment from report."
    )
    geo_section = (
        _format_geo_analysis(geo_analysis)
        if geo_analysis else "Not available — draw on web research for geopolitical context."
    )
    specialist_section = (
        _format_sector_analysis(sector_analysis)
        if sector_analysis else "Not available — draw on web research for technology context."
    )

    hypothesis_input = (
        f"User question: {question}\n\n"
        f"Sector: {plan.sector} | Companies analysed: {', '.join(plan.tickers)}\n\n"
        f"EDA findings:\n{eda_findings.model_dump_json(indent=2)}\n\n"
        f"Data summary: {data_bundle.summary}"
        f"{research_section}\n\n"
        f"━━ LIVE VALUATION DATA (from market data — use these real multiples in Section 6) ━━\n"
        f"{valuation_section}\n\n"
        f"━━ MARKET SENTIMENT (from news analysis — use in Investment Summary section) ━━\n"
        f"{sentiment_section}\n\n"
        f"━━ GEOPOLITICAL ANALYSIS (from Geopolitical Advisor — use for Section 5) ━━\n"
        f"{geo_section}\n\n"
        f"━━ SECTOR SPECIALIST ANALYSIS (from domain expert — use for Section 4) ━━\n"
        f"{specialist_section}"
    )
    report = await _run_agent(hypothesis_agent, hypothesis_input, HypothesisReport, _status_sink=rate_sink)
    for msg in _drain(rate_sink):
        yield "status", msg

    # ── Post-process artifact_paths: use side-channel path from save_report() ─
    recorded_path = get_last_saved_report_path()
    valid_md_in_json = [p for p in report.artifact_paths if p.endswith(".md") and Path(p).exists()]

    if recorded_path and recorded_path not in report.artifact_paths:
        # Agent put wrong string in artifact_paths — correct it using the recorded path
        report = report.model_copy(update={
            "artifact_paths": [recorded_path] + [p for p in report.artifact_paths if not p.endswith(".md")]
        })
        logger.info("artifact_paths corrected via side-channel: %s", recorded_path)
    elif not valid_md_in_json and not recorded_path:
        # save_report() was never called — last-resort scan
        candidates = sorted(Path("artifacts").glob("report_*.md"),
                            key=lambda f: f.stat().st_mtime, reverse=True)
        if candidates:
            fallback = str(candidates[0].relative_to(Path("artifacts").parent))
            report = report.model_copy(update={
                "artifact_paths": [fallback] + [p for p in report.artifact_paths if not p.endswith(".md")]
            })
            logger.warning("artifact_paths: last-resort fallback to %s", fallback)

    chart_paths: list[str] = []
    for finding in eda_findings.findings:
        if finding.artifact_path:
            chart_paths.append(finding.artifact_path)
    chart_paths.extend(report.artifact_paths)

    # Store context for Q&A agent use
    # Find the markdown report path from artifact_paths (exclude chart PNGs)
    md_report_path = next(
        (p for p in report.artifact_paths if p.endswith(".md")),
        None,
    )
    global _last_run_context
    _last_run_context = {
        "sector": plan.sector,
        "tickers": plan.tickers,
        "report_path": md_report_path,
        "geo_analysis": geo_analysis,
        "sector_analysis": sector_analysis,
    }

    logger.info("STEP_TIMING step=7 name=Hypothesis elapsed=%.1f", time.perf_counter() - _t0)

    yield "progress", {
        "step": 7, "total_steps": 7, "step_name": "Complete",
        "elapsed_seconds": int(time.perf_counter() - _t0),
        "estimated_remaining_seconds": 0, "pct": 100,
    }
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
