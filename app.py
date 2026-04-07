"""FastAPI application — serves the frontend and exposes the analysis API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.orchestrator import run_analysis

app = FastAPI(title="Multi-Agent Data Analyst")

FRONTEND_DIR = Path(__file__).parent / "frontend"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Serve generated artifacts (charts, reports) as static files
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


class AnalysisRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    hypothesis: str
    evidence: list[dict]
    narrative: str
    artifact_paths: list[str]
    confidence: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (FRONTEND_DIR / "index.html").read_text()
    return HTMLResponse(content=html)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest) -> AnalysisResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await run_analysis(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # result is a HypothesisReport (structured output from the Hypothesis agent)
    return AnalysisResponse(
        hypothesis=result.hypothesis,
        evidence=[e.model_dump() for e in result.evidence],
        narrative=result.narrative,
        artifact_paths=result.artifact_paths,
        confidence=result.confidence,
    )


@app.get("/artifacts/list")
def list_artifacts() -> dict:
    """List all generated artifact files (charts, reports)."""
    files = [
        {"name": f.name, "url": f"/artifacts/{f.name}"}
        for f in ARTIFACTS_DIR.iterdir()
        if f.suffix in {".png", ".csv", ".md"} and not f.name.startswith("_")
    ]
    return {"artifacts": files}
