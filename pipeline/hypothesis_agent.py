"""Hypothesis Agent — synthesizes EDA findings into a grounded report.

Step 3: Hypothesize. This agent:
- Receives EDAFindings from the EDA agent
- Forms a hypothesis grounded in specific data points (not model weights)
- Cites evidence with specific numbers, percentages, time ranges
- Generates a visualization summary if not already done
- Writes a markdown report to disk as a persistent artifact
- Returns a structured HypothesisReport

Grab bag: Artifacts (markdown report + charts to disk), Structured Output
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.code_executor import execute_python
from models.schemas import HypothesisReport

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def _make_model() -> LitellmModel:
    return LitellmModel(
        model=LITELLM_MODEL_ID,
        extra_kwargs={
            "vertex_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "vertex_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        },
    )


HYPOTHESIS_PROMPT = """\
You are the Hypothesis agent in a multi-agent Sector Analyst system.

You receive:
- The user's original sector/company question
- EDAFindings from the EDA agent: computed metrics, trends, anomalies, chart paths

Your job is to form and communicate a financial hypothesis — the analyst's
"so what?" grounded in SEC filing data, not model weights.

RULES:
- Every claim MUST cite a specific data point from EDAFindings (a number,
  percentage, time range, or company comparison). Do not rely on prior knowledge.
- Do not hallucinate figures. If a metric isn't in the EDA findings, don't cite it.
- If the evidence is weak or limited, rate confidence 'low' and say why.
- Write a full narrative structured like an analyst memo:
    1. Executive Summary (2-3 sentences)
    2. Key Findings (bullet points with specific numbers)
    3. Hypothesis & Reasoning (what the data suggests and why)
    4. Risks / Alternative Explanations
    5. Conclusion

TOOLS:
- save_report: persist a markdown memo to disk (always call this)
- run_python: generate a final summary chart if needed

OUTPUT — HypothesisReport:
- title: e.g. "Semiconductor Sector: Margin Divergence 2020–2024"
- hypothesis: 1-2 sentence main claim grounded in the data
- evidence: list of EvidencePoints (claim, data_point, source)
- narrative: full analyst memo
- artifact_paths: all saved files (report + any charts)
- confidence: 'high', 'medium', or 'low'
"""


@function_tool(strict_mode=False)
def save_report(title: str, content: str) -> str:
    """
    Save a markdown report to disk as a persistent artifact.
    Returns the file path of the saved report.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    filename = f"report_{uuid.uuid4().hex[:8]}.md"
    path = ARTIFACTS_DIR / filename
    path.write_text(f"# {title}\n\n{content}")
    return str(path.relative_to(ARTIFACTS_DIR.parent))


@function_tool(strict_mode=False)
def run_python(code: str) -> dict:
    """Execute Python code for a final visualization or computation."""
    return execute_python(code)


def build_hypothesis_agent() -> Agent:
    return Agent(
        name="Hypothesis",
        model=_make_model(),
        instructions=HYPOTHESIS_PROMPT,
        tools=[save_report, run_python],
        output_type=HypothesisReport,
    )
