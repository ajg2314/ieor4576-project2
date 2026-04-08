"""Pydantic schemas for structured data flow between agents."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DataBundle(BaseModel):
    """Structured output from the Collector agent."""
    source: str = Field(description="Name/URL of the primary data source")
    retrieval_method: str = Field(description="'api', 'sql', 'web', or 'rag'")
    records: list[dict[str, Any]] = Field(default_factory=list, description="Retrieved data rows/records")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source metadata (units, time range, etc.)")
    summary: str = Field(description="Short natural-language description of what was retrieved and why")


class EDAFinding(BaseModel):
    """A single finding produced by one EDA tool."""
    tool_name: str
    description: str = Field(description="What this finding shows")
    value: Any = Field(description="The computed result (number, table, string)")
    artifact_path: str | None = Field(default=None, description="Path to generated chart/file, if any")


class EDAFindings(BaseModel):
    """Structured output from the EDA agent."""
    findings: list[EDAFinding]
    key_insight: str = Field(description="The single most important pattern or anomaly discovered")
    recommended_hypothesis_direction: str = Field(
        description="Suggested direction for the hypothesis agent based on EDA results"
    )


class EvidencePoint(BaseModel):
    """A specific data point cited in the hypothesis."""
    claim: str
    data_point: str = Field(description="The specific number, percentage, or observation")
    source: str = Field(description="Which tool/dataset this came from")


class HypothesisReport(BaseModel):
    """Structured output from the Hypothesis agent — the final deliverable."""
    title: str
    hypothesis: str = Field(description="The main hypothesis statement in one or two sentences")
    evidence: list[EvidencePoint]
    narrative: str = Field(description="Full natural-language report with reasoning")
    artifact_paths: list[str] = Field(default_factory=list, description="Paths to charts/reports saved to disk")
    confidence: str = Field(description="'high', 'medium', or 'low' — justified by evidence strength")
