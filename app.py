"""FastAPI application — serves the frontend and exposes the analysis API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import agents
agents.set_tracing_disabled(True)

# LiteLLM reads these env vars for Vertex AI authentication
os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
os.environ.setdefault("VERTEXAI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.orchestrator import run_analysis_with_status, get_last_run_context
from pipeline.qa_agent import build_qa_agent
from models.schemas import QAResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Sector Analyst")

FRONTEND_DIR = Path(__file__).parent / "frontend"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Mount at /files/ so /api/artifacts/list route is not shadowed
app.mount("/files", StaticFiles(directory=str(ARTIFACTS_DIR)), name="files")


class AnalysisRequest(BaseModel):
    question: str
    prior_context: str | None = None


class QARequest(BaseModel):
    question: str
    report_path: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (FRONTEND_DIR / "index.html").read_text()
    return HTMLResponse(content=html)


@app.get("/api/artifacts/list")
def list_artifacts() -> dict:
    """List all generated artifact files (charts, reports)."""
    files = [
        {"name": f.name, "url": f"/files/{f.name}"}
        for f in sorted(ARTIFACTS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        if f.suffix in {".png", ".csv", ".md"} and not f.name.startswith("_")
    ]
    return {"artifacts": files}


@app.post("/analyze/stream")
async def analyze_stream(req: AnalysisRequest) -> StreamingResponse:
    """SSE endpoint — streams status events then the final result."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_stream():
        try:
            async for event_type, payload in run_analysis_with_status(req.question, req.prior_context):
                data = json.dumps({"type": event_type, "payload": payload})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # let the event loop flush
        except Exception as e:
            logger.error("Pipeline error: %s", traceback.format_exc())
            error_data = json.dumps({"type": "error", "payload": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _load_report(report_path: str | None) -> str:
    """Load report text from a path, falling back to the most recent artifact."""
    if report_path:
        p = Path(report_path)
        if not p.is_absolute():
            p = Path(__file__).parent / report_path
        if p.exists():
            return p.read_text()
    # Fall back: most recently modified report in artifacts/
    reports = sorted(
        ARTIFACTS_DIR.glob("report_*.md"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return reports[0].read_text() if reports else ""


def _extract_qa_text(result) -> str:
    from agents import ItemHelpers
    from agents.items import MessageOutputItem
    raw = result.final_output
    if raw and str(raw).strip():
        return str(raw)
    for item in reversed(result.new_items):
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text.strip():
                return text
    return ""


@app.post("/qa/stream")
async def qa_stream(req: QARequest) -> StreamingResponse:
    """SSE endpoint — answers a follow-up question about the most recent analyst report."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        try:
            from agents import Runner
            import re

            ctx = get_last_run_context()
            report_text = _load_report(req.report_path or ctx.get("report_path"))

            if not report_text and not ctx:
                error_data = json.dumps({
                    "type": "error",
                    "payload": "No analysis available. Run /analyze/stream first.",
                })
                yield f"data: {error_data}\n\n"
                return

            qa_agent = build_qa_agent(
                full_report=report_text,
                sector=ctx.get("sector", ""),
                tickers=ctx.get("tickers", []),
                geo_analysis=ctx.get("geo_analysis"),
                sector_analysis=ctx.get("sector_analysis"),
            )

            result = await Runner.run(qa_agent, input=req.question)
            raw = _extract_qa_text(result)

            # Parse or fall back to plain answer
            try:
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
                response = QAResponse.model_validate_json(cleaned)
            except Exception:
                response = QAResponse(
                    question=req.question,
                    answer=raw or "No response generated.",
                    answer_type="answer",
                )

            data = json.dumps({"type": "qa_result", "payload": response.model_dump()})
            yield f"data: {data}\n\n"

        except Exception as e:
            logger.error("QA error: %s", traceback.format_exc())
            error_data = json.dumps({"type": "error", "payload": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
