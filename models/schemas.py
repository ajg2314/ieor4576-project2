"""Pydantic schemas for structured data flow between agents."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SectorPlan(BaseModel):
    """Structured output from the Planner agent — scope and company list for a research question."""
    sector: str = Field(description="Sector or industry name (e.g. 'Semiconductors', 'Cloud Software')")
    expanded_query: str = Field(description="Enriched version of the user's question with full context")
    tickers: list[str] = Field(description="10-20 ticker symbols covering the sector's most important companies")
    rationale: str = Field(description="Brief explanation of why these companies were selected")
    focus_metrics: list[str] = Field(
        default_factory=lambda: ["revenue", "gross_profit", "operating_income", "net_income", "rd_expense"],
        description="Financial metrics most relevant to answering the question",
    )


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


class ResearchSource(BaseModel):
    """A single source consulted during qualitative research."""
    title: str
    url: str
    snippet: str = Field(description="Key excerpt or summary from this source")


class ResearchContext(BaseModel):
    """Structured output from the Research agent — qualitative sector intelligence."""
    sector: str
    technology_context: str = Field(
        description="Key technology trends, product cycles, and what the market is betting on for 3-5 years"
    )
    market_context: str = Field(
        description="Supply/demand dynamics, competitive landscape, who is gaining/losing share and why"
    )
    geopolitical_context: str = Field(
        default="",
        description="Government policies, trade actions, export controls, geographic risks, subsidies — with specific names, dates, and impact on companies"
    )
    expert_sentiment: str = Field(
        description="Specific analyst opinions, price targets, earnings call themes, and investor perspectives"
    )
    key_risks: list[str] = Field(
        description="Top 3-5 specific risks with mechanism of impact"
    )
    qualitative_insights: list[str] = Field(
        description="Forward-looking insights: what the market is pricing in, competitive dynamics, policy factors shaping the next 2-3 years"
    )
    sources_consulted: list[ResearchSource] = Field(
        default_factory=list,
        description="Web sources, analyst reports, and articles reviewed"
    )


class HypothesisReport(BaseModel):
    """Structured output from the Hypothesis agent — the final deliverable."""
    title: str
    hypothesis: str = Field(description="The main hypothesis statement in one or two sentences")
    evidence: list[EvidencePoint]
    narrative: str = Field(description="Full natural-language report with reasoning")
    artifact_paths: list[str] = Field(default_factory=list, description="Paths to charts/reports saved to disk")
    confidence: str = Field(description="'high', 'medium', or 'low' — justified by evidence strength")


# ── Phase 1 expansion schemas ─────────────────────────────────────────────────

class PeerInfo(BaseModel):
    """Validated, enriched data for one peer company from market data sources."""
    ticker: str
    company_name: str = Field(default="")
    market_cap_b: float | None = Field(default=None, description="Market cap in billions USD")
    sector: str = Field(default="")
    industry: str = Field(default="")
    valid: bool = Field(default=True, description="False if ticker failed validation or market cap < threshold")


class PeerList(BaseModel):
    """Output of peer_discovery: validated, sorted, enriched ticker universe."""
    sector: str
    peers: list[PeerInfo]
    tickers: list[str] = Field(description="Valid ticker symbols sorted by market cap descending")
    selection_rationale: str = Field(default="")


class ValuationMetrics(BaseModel):
    """Per-company valuation data fetched from live market data sources."""
    ticker: str
    company_name: str = Field(default="")
    current_price: float | None = None
    market_cap_b: float | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None
    ev_ebitda: float | None = None
    ev_revenue: float | None = None
    price_to_book: float | None = None
    ytd_return_pct: float | None = None
    analyst_recommendation: str | None = None
    price_target: float | None = None


class ValuationContext(BaseModel):
    """Output of ValuationAgent: sector comp table + median multiples + interpretation."""
    sector: str
    as_of_date: str = Field(description="ISO date string of when data was fetched")
    metrics: list[ValuationMetrics]
    sector_median_pe: float | None = None
    sector_median_ev_ebitda: float | None = None
    summary: str = Field(description="LLM interpretation of relative valuations across peers")


class SentimentContext(BaseModel):
    """Output of SentimentAgent: market sentiment synthesis from news and analyst commentary."""
    sector: str
    overall_sentiment: str = Field(description="'bullish', 'neutral', or 'bearish'")
    sentiment_score: float = Field(description="Float from -1.0 (bearish) to 1.0 (bullish)")
    key_themes: list[str] = Field(default_factory=list, description="3-5 recurring themes from recent coverage")
    recent_headlines: list[str] = Field(default_factory=list, description="Representative recent headlines")
    earnings_highlights: list[str] = Field(default_factory=list, description="Key earnings call themes")
    summary: str = Field(description="2-3 sentence synthesis of market sentiment")


# ── Phase 2 expansion schemas ─────────────────────────────────────────────────

class GeopoliticalAnalysis(BaseModel):
    """Output of GeopoliticalAdvisor: sector-specific geopolitical risk and policy analysis."""
    sector: str
    key_policies: list[str] = Field(
        default_factory=list,
        description="Named, dated policies with quantified impact. E.g. 'US BIS Oct 2024: restricts AI chips >1800 TFLOPS'"
    )
    company_exposures: list[dict] = Field(
        default_factory=list,
        description="Per-company exposure. E.g. [{'ticker': 'NVDA', 'exposure': 'high', 'mechanism': 'China revenue ~17% at risk'}]"
    )
    geographic_risks: list[str] = Field(default_factory=list)
    policy_tailwinds: list[str] = Field(default_factory=list, description="Subsidies and industrial policy benefits")
    tail_risks: list[str] = Field(default_factory=list)
    summary: str = Field(description="2-3 paragraphs ready to paste into report Section 5 (Geopolitical & Macro)")


class SectorAnalysis(BaseModel):
    """Output of SectorSpecialist: domain expert analysis of sector technology and competitive dynamics."""
    sector: str
    specialist_type: str = Field(description="'tech' | 'biomedical' | 'energy' | 'financials' | 'general'")
    technology_sota: str = Field(description="Current state of the art — what is actually happening in products and research")
    competitive_dynamics: str = Field(description="Who is winning, how moats are built or eroded")
    forward_thesis: str = Field(description="What the market is betting on for the next 3-5 years")
    key_disruptions: list[str] = Field(default_factory=list, description="Emerging threats or structural opportunities")
    summary: str = Field(description="2-3 paragraphs ready to paste into report Section 4 (Technology & Innovation)")


class QAResponse(BaseModel):
    """Output of QA agent — answer to a user follow-up question or summarize command."""
    question: str
    answer: str
    answer_type: str = Field(default="answer", description="'summary' | 'answer'")
    artifact_paths: list[str] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
