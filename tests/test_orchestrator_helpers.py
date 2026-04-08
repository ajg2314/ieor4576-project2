"""Tests for orchestrator helper functions and the SectorPlan schema."""
import pytest
from models.schemas import SectorPlan, DataBundle
from pipeline.orchestrator import _compact_for_eda, _needs_refinement, EDA_MAX_YEARS
from models.schemas import EDAFindings, EDAFinding


# ── SectorPlan schema ─────────────────────────────────────────────────────────

class TestSectorPlan:
    def test_valid_plan(self):
        plan = SectorPlan(
            sector="Semiconductors",
            expanded_query="Analyze revenue and R&D trends across the semiconductor sector",
            tickers=["NVDA", "AMD", "INTC", "TSM", "ASML", "AMAT"],
            rationale="Covers GPU, CPU, foundry, and equipment leaders",
        )
        assert plan.sector == "Semiconductors"
        assert "NVDA" in plan.tickers
        assert "TSM" in plan.tickers

    def test_default_focus_metrics(self):
        plan = SectorPlan(
            sector="Cloud",
            expanded_query="Cloud revenue growth",
            tickers=["MSFT", "AMZN", "GOOG"],
            rationale="Top 3 hyperscalers",
        )
        assert "revenue" in plan.focus_metrics
        assert "gross_profit" in plan.focus_metrics

    def test_custom_focus_metrics(self):
        plan = SectorPlan(
            sector="Biotech",
            expanded_query="Biotech R&D analysis",
            tickers=["AMGN", "GILD"],
            rationale="Large-cap biotech",
            focus_metrics=["rd_expense", "net_income"],
        )
        assert plan.focus_metrics == ["rd_expense", "net_income"]


# ── _compact_for_eda ──────────────────────────────────────────────────────────

def _make_bundle(records):
    return DataBundle(
        source="SEC EDGAR",
        retrieval_method="api",
        records=records,
        summary="test",
    )


def _make_record(ticker, year, form="10-K", metric="revenue", value=1e9):
    return {
        "ticker": ticker,
        "company": f"{ticker} Inc",
        "period": f"{year}-12-31",
        "fiscal_year": str(year),
        "form": form,
        "metric": metric,
        "value": value,
        "value_billions": value / 1e9,
    }


class TestCompactForEda:
    def test_filters_to_annual_only(self):
        records = [
            _make_record("NVDA", 2023, "10-K"),
            _make_record("NVDA", 2023, "10-Q"),
            _make_record("NVDA", 2022, "10-K"),
            _make_record("NVDA", 2022, "10-Q"),
        ]
        compact = _compact_for_eda(_make_bundle(records))
        assert all(r["form"] == "10-K" for r in compact.records)
        assert len(compact.records) == 2

    def test_limits_to_eda_max_years(self):
        records = [
            _make_record("NVDA", year, "10-K")
            for year in range(2010, 2025)  # 15 years
        ]
        compact = _compact_for_eda(_make_bundle(records))
        years = set(r["fiscal_year"] for r in compact.records)
        assert len(years) <= EDA_MAX_YEARS

    def test_keeps_most_recent_years(self):
        records = [
            _make_record("NVDA", year, "10-K")
            for year in range(2015, 2025)
        ]
        compact = _compact_for_eda(_make_bundle(records))
        years = sorted(set(r["fiscal_year"] for r in compact.records))
        # Should keep the most recent EDA_MAX_YEARS years
        assert max(years) == "2024"

    def test_falls_back_to_all_when_no_annual(self):
        records = [
            _make_record("NVDA", 2023, "10-Q"),
            _make_record("NVDA", 2022, "10-Q"),
        ]
        compact = _compact_for_eda(_make_bundle(records))
        assert len(compact.records) == 2  # fallback keeps all

    def test_multiple_companies_preserved(self):
        records = [
            _make_record("NVDA", 2023, "10-K"),
            _make_record("AMD", 2023, "10-K"),
            _make_record("INTC", 2023, "10-K"),
        ]
        compact = _compact_for_eda(_make_bundle(records))
        tickers = {r["ticker"] for r in compact.records}
        assert tickers == {"NVDA", "AMD", "INTC"}

    def test_source_and_summary_preserved(self):
        bundle = _make_bundle([_make_record("NVDA", 2023)])
        bundle = DataBundle(
            source="custom source",
            retrieval_method="api",
            records=bundle.records,
            summary="custom summary",
        )
        compact = _compact_for_eda(bundle)
        assert compact.source == "custom source"
        assert compact.summary == "custom summary"


# ── _needs_refinement ─────────────────────────────────────────────────────────

class TestNeedsRefinement:
    def _findings(self, direction: str) -> EDAFindings:
        return EDAFindings(
            findings=[EDAFinding(tool_name="stats_tool", description="x", value=1)],
            key_insight="test",
            recommended_hypothesis_direction=direction,
        )

    def test_detects_missing_data(self):
        assert _needs_refinement(self._findings("missing R&D data for TSMC"))

    def test_detects_insufficient(self):
        assert _needs_refinement(self._findings("data is insufficient for conclusion"))

    def test_detects_need_more(self):
        assert _needs_refinement(self._findings("need more historical data"))

    def test_no_refinement_when_complete(self):
        assert not _needs_refinement(self._findings("focus on GPU margin expansion at NVDA"))

    def test_no_refinement_empty_direction(self):
        assert not _needs_refinement(self._findings(""))
