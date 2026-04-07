"""Tests for pipeline/orchestrator.py — parsing, refinement logic, text extraction."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.orchestrator import _parse_text, _needs_refinement, _extract_text
from models.schemas import DataBundle, EDAFindings, EDAFinding, HypothesisReport, EvidencePoint


# ── _parse_text ───────────────────────────────────────────────────────────────

class TestParseText:
    def _valid_bundle_json(self):
        return '{"source":"SEC","retrieval_method":"api","records":[],"metadata":{},"summary":"test"}'

    def test_valid_json(self):
        bundle = _parse_text(self._valid_bundle_json(), DataBundle)
        assert bundle.source == "SEC"

    def test_strips_markdown_fences(self):
        raw = f"```json\n{self._valid_bundle_json()}\n```"
        bundle = _parse_text(raw, DataBundle)
        assert bundle.source == "SEC"

    def test_strips_plain_code_fence(self):
        raw = f"```\n{self._valid_bundle_json()}\n```"
        bundle = _parse_text(raw, DataBundle)
        assert bundle.source == "SEC"

    def test_extracts_json_from_mixed_text(self):
        raw = f"Here is my output:\n{self._valid_bundle_json()}\nDone."
        bundle = _parse_text(raw, DataBundle)
        assert bundle.source == "SEC"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="no text output"):
            _parse_text("", DataBundle)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="no text output"):
            _parse_text("   \n\t  ", DataBundle)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_text("this is not json at all", DataBundle)

    def test_wrong_schema_raises(self):
        # Valid JSON but wrong schema
        with pytest.raises(ValueError):
            _parse_text('{"foo": "bar"}', DataBundle)

    def test_eda_findings_parsed(self):
        raw = '''{
            "findings": [{"tool_name": "stats_tool", "description": "Revenue grew", "value": 12.5, "artifact_path": null}],
            "key_insight": "NVDA margin tripled",
            "recommended_hypothesis_direction": "Focus on data center"
        }'''
        findings = _parse_text(raw, EDAFindings)
        assert findings.key_insight == "NVDA margin tripled"
        assert len(findings.findings) == 1

    def test_hypothesis_report_parsed(self):
        raw = '''{
            "title": "NVDA Dominance",
            "hypothesis": "NVDA leads on margin.",
            "evidence": [{"claim": "Margin grew", "data_point": "61%", "source": "SEC EDGAR"}],
            "narrative": "Full text...",
            "artifact_paths": ["artifacts/chart.png"],
            "confidence": "high"
        }'''
        report = _parse_text(raw, HypothesisReport)
        assert report.confidence == "high"
        assert len(report.evidence) == 1


# ── _needs_refinement ─────────────────────────────────────────────────────────

class TestNeedsRefinement:
    def _findings(self, direction: str) -> EDAFindings:
        return EDAFindings(
            findings=[],
            key_insight="some insight",
            recommended_hypothesis_direction=direction,
        )

    def test_missing_data_triggers_refinement(self):
        assert _needs_refinement(self._findings("There is missing data for INTC"))

    def test_insufficient_triggers(self):
        assert _needs_refinement(self._findings("Data is insufficient for a strong hypothesis"))

    def test_need_more_triggers(self):
        assert _needs_refinement(self._findings("We need more quarterly data"))

    def test_additional_data_triggers(self):
        assert _needs_refinement(self._findings("Requires additional data from EDGAR"))

    def test_no_data_triggers(self):
        assert _needs_refinement(self._findings("No data found for the requested period"))

    def test_case_insensitive(self):
        assert _needs_refinement(self._findings("MISSING data for AMD"))

    def test_solid_findings_no_refinement(self):
        assert not _needs_refinement(self._findings("Focus on NVDA margin expansion trend"))

    def test_empty_direction_no_refinement(self):
        assert not _needs_refinement(self._findings(""))

    def test_strong_hypothesis_no_refinement(self):
        assert not _needs_refinement(self._findings(
            "Hypothesis: NVDA operating margin expanded from 20% to 61% — focus on that"
        ))


# ── _extract_text ─────────────────────────────────────────────────────────────

class TestExtractText:
    def _make_result(self, final_output=None, items=None):
        from agents.items import MessageOutputItem
        result = MagicMock()
        result.final_output = final_output
        result.new_items = items or []
        return result

    def test_returns_final_output_when_present(self):
        result = self._make_result(final_output='{"key": "value"}')
        assert _extract_text(result) == '{"key": "value"}'

    def test_skips_empty_final_output(self):
        from agents.items import MessageOutputItem
        # Build a mock MessageOutputItem
        mock_item = MagicMock(spec=MessageOutputItem)
        with patch("pipeline.orchestrator.ItemHelpers") as mock_helpers:
            mock_helpers.text_message_output.return_value = "fallback text"
            result = self._make_result(final_output="", items=[mock_item])
            text = _extract_text(result)
            assert text == "fallback text"

    def test_returns_empty_when_no_items(self):
        result = self._make_result(final_output="", items=[])
        text = _extract_text(result)
        assert text == ""

    def test_none_final_output_falls_through(self):
        result = self._make_result(final_output=None, items=[])
        text = _extract_text(result)
        assert text == ""
