"""Tests for Pydantic schemas in models/schemas.py."""
import pytest
from pydantic import ValidationError
from models.schemas import DataBundle, EDAFinding, EDAFindings, EvidencePoint, HypothesisReport


# ── DataBundle ────────────────────────────────────────────────────────────────

class TestDataBundle:
    def test_valid_minimal(self):
        bundle = DataBundle(
            source="SEC EDGAR",
            retrieval_method="api",
            records=[{"ticker": "AAPL", "period": "2023", "metric": "revenue", "value": 383285000000}],
            summary="Retrieved Apple revenue data",
        )
        assert bundle.source == "SEC EDGAR"
        assert len(bundle.records) == 1
        assert bundle.metadata == {}  # default

    def test_valid_with_metadata(self):
        bundle = DataBundle(
            source="SEC EDGAR XBRL",
            retrieval_method="api",
            records=[],
            metadata={"companies": ["AAPL", "MSFT"], "time_range": "2020-2024"},
            summary="Sector data",
        )
        assert bundle.metadata["companies"] == ["AAPL", "MSFT"]

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            DataBundle(source="SEC EDGAR", retrieval_method="api", records=[])
            # missing 'summary'

    def test_records_accepts_arbitrary_dicts(self):
        bundle = DataBundle(
            source="test", retrieval_method="sql",
            records=[{"a": 1, "b": "two", "c": [1, 2, 3]}],
            summary="test",
        )
        assert bundle.records[0]["c"] == [1, 2, 3]

    def test_json_roundtrip(self):
        bundle = DataBundle(
            source="SEC EDGAR", retrieval_method="api",
            records=[{"ticker": "NVDA", "value": 44870000000}],
            summary="NVDA revenue",
        )
        json_str = bundle.model_dump_json()
        restored = DataBundle.model_validate_json(json_str)
        assert restored.source == bundle.source
        assert restored.records == bundle.records


# ── EDAFindings ───────────────────────────────────────────────────────────────

class TestEDAFindings:
    def _finding(self, **kwargs):
        defaults = {"tool_name": "stats_tool", "description": "Revenue growth", "value": 12.5}
        return EDAFinding(**{**defaults, **kwargs})

    def test_valid(self):
        findings = EDAFindings(
            findings=[self._finding()],
            key_insight="NVDA revenue grew 122% YoY",
            recommended_hypothesis_direction="Focus on data center segment",
        )
        assert findings.key_insight == "NVDA revenue grew 122% YoY"

    def test_finding_with_artifact_path(self):
        f = EDAFinding(
            tool_name="run_python",
            description="Revenue trend chart",
            value=None,
            artifact_path="artifacts/revenue_trend.png",
        )
        assert f.artifact_path == "artifacts/revenue_trend.png"

    def test_finding_artifact_path_optional(self):
        f = self._finding()
        assert f.artifact_path is None

    def test_empty_findings_list_valid(self):
        # Empty findings is technically valid — schema doesn't enforce minimum
        findings = EDAFindings(
            findings=[],
            key_insight="No data found",
            recommended_hypothesis_direction="Need more data",
        )
        assert findings.findings == []

    def test_json_roundtrip(self):
        findings = EDAFindings(
            findings=[self._finding(value={"mean": 100, "std": 10})],
            key_insight="Margins compressed 5pp",
            recommended_hypothesis_direction="Investigate cost drivers",
        )
        restored = EDAFindings.model_validate_json(findings.model_dump_json())
        assert restored.key_insight == findings.key_insight


# ── HypothesisReport ──────────────────────────────────────────────────────────

class TestHypothesisReport:
    def _evidence(self, **kwargs):
        defaults = {
            "claim": "Revenue grew", "data_point": "122% YoY",
            "source": "SEC EDGAR 10-K 2023",
        }
        return EvidencePoint(**{**defaults, **kwargs})

    def test_valid(self):
        report = HypothesisReport(
            title="NVDA Margin Expansion",
            hypothesis="NVDA's operating margin tripled from 20% to 61% over 4 years.",
            evidence=[self._evidence()],
            narrative="Full analysis text here...",
            confidence="high",
        )
        assert report.confidence == "high"
        assert report.artifact_paths == []  # default

    def test_confidence_values(self):
        for conf in ("high", "medium", "low"):
            r = HypothesisReport(
                title="t", hypothesis="h", evidence=[],
                narrative="n", confidence=conf,
            )
            assert r.confidence == conf

    def test_artifact_paths(self):
        report = HypothesisReport(
            title="t", hypothesis="h", evidence=[],
            narrative="n", confidence="low",
            artifact_paths=["artifacts/chart.png", "artifacts/report.md"],
        )
        assert len(report.artifact_paths) == 2

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            HypothesisReport(title="t", hypothesis="h")  # missing narrative, confidence, evidence

    def test_json_roundtrip(self):
        report = HypothesisReport(
            title="Sector Analysis",
            hypothesis="Cloud margins expanded.",
            evidence=[self._evidence(), self._evidence(claim="Costs fell", data_point="3pp")],
            narrative="Long narrative...",
            confidence="medium",
            artifact_paths=["artifacts/chart.png"],
        )
        restored = HypothesisReport.model_validate_json(report.model_dump_json())
        assert len(restored.evidence) == 2
        assert restored.artifact_paths == ["artifacts/chart.png"]
