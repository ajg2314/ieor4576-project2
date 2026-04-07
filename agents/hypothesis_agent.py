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
You are the Hypothesis agent in a multi-agent data analysis system.

You receive:
- The user's original analytics question
- EDAFindings from the EDA agent (statistics, patterns, anomalies, chart paths)

Your job is to form and communicate a data-grounded hypothesis. This is the
"so what?" of the analysis.

Rules:
- Every claim must cite a specific data point from the EDAFindings (a number,
  percentage, time range, or group comparison). Do not rely on prior knowledge.
- Do not hallucinate data. If the evidence is weak, say so and rate confidence low.
- Write a full narrative that explains your reasoning step by step.
- Use save_report to persist a markdown report to disk.
- Use run_python if you need a final summary visualization not already in EDA.

Your output must be a structured HypothesisReport with:
- title: short descriptive title
- hypothesis: 1-2 sentence main claim
- evidence: list of specific data points cited
- narrative: full report with reasoning
- artifact_paths: list of files saved
- confidence: 'high', 'medium', or 'low' with justification in the narrative
"""


@function_tool
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


@function_tool
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
