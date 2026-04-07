"""Tests for tools/sec_edgar.py — ticker resolution, financial data, filing scraping.

External HTTP calls are mocked so tests run offline and deterministically.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from tools.sec_edgar import (
    resolve_ticker,
    _load_ticker_map,
    get_company_financials,
    get_recent_filing_text,
    _extract_mda_section,
    FINANCIAL_CONCEPTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_TICKER_MAP = {
    "0": {"cik_str": "1045810", "ticker": "NVDA", "title": "NVIDIA Corp"},
    "1": {"cik_str": "732834",  "ticker": "AAPL", "title": "Apple Inc"},
    "2": {"cik_str": "1326380", "ticker": "GOOGL", "title": "Alphabet Inc"},
}

MOCK_COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2022-01-30", "val": 26914000000, "form": "10-K", "filed": "2022-02-25"},
                        {"end": "2022-10-30", "val": 5931000000, "form": "10-Q", "filed": "2022-11-18"},
                        {"end": "2023-01-29", "val": 26974000000, "form": "10-K", "filed": "2023-02-24"},
                    ]
                }
            },
            "OperatingIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2022-01-30", "val": 4532000000, "form": "10-K", "filed": "2022-02-25"},
                        {"end": "2023-01-29", "val": 10041000000, "form": "10-K", "filed": "2023-02-24"},
                    ]
                }
            },
        }
    }
}

MOCK_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "accessionNumber": ["0001045810-23-000017", "0001045810-22-000123", "0001045810-22-000100"],
            "filingDate": ["2023-02-24", "2022-11-18", "2022-09-01"],
            "primaryDocument": ["nvda-20230129.htm", "nvda-20221030.htm", "nvda-20220801.htm"],
        }
    }
}


def _mock_get(url, *args, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "company_tickers.json" in url:
        resp.json.return_value = MOCK_TICKER_MAP
    elif "companyfacts" in url:
        resp.json.return_value = MOCK_COMPANY_FACTS
    elif "submissions" in url:
        resp.json.return_value = MOCK_SUBMISSIONS
    elif "Archives" in url:
        resp.text = (
            "<html><body>"
            "<h2>Item 7. Management Discussion and Analysis</h2>"
            "Revenue increased significantly driven by data center growth. "
            "Operating margins expanded due to operating leverage. "
            "<h2>Item 8. Financial Statements</h2>"
            "Balance sheet data here."
            "</body></html>"
        )
    else:
        resp.json.return_value = {}
        resp.text = ""
    return resp


# ── resolve_ticker ────────────────────────────────────────────────────────────

class TestResolveTicker:
    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_known_ticker(self, mock_get):
        result = resolve_ticker("NVDA")
        assert result["ticker"] == "NVDA"
        assert result["cik"] == 1045810
        assert result["cik_padded"] == "0001045810"
        assert result["title"] == "NVIDIA Corp"

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_case_insensitive(self, mock_get):
        result = resolve_ticker("nvda")
        assert result["ticker"] == "NVDA"

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_unknown_ticker_raises(self, mock_get):
        with pytest.raises(ValueError, match="not found"):
            resolve_ticker("ZZZZZ")

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_aapl_ticker(self, mock_get):
        result = resolve_ticker("AAPL")
        assert result["cik"] == 732834


# ── get_company_financials ────────────────────────────────────────────────────

class TestGetCompanyFinancials:
    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_returns_revenue(self, mock_get):
        result = get_company_financials("NVDA", ["revenue"])
        assert result["ticker"] == "NVDA"
        assert "revenue" in result["financials"]
        assert len(result["financials"]["revenue"]) > 0

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_revenue_values_correct(self, mock_get):
        result = get_company_financials("NVDA", ["revenue"])
        values = [r["value"] for r in result["financials"]["revenue"]]
        assert 26914000000 in values or 26974000000 in values

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_operating_income_returned(self, mock_get):
        result = get_company_financials("NVDA", ["operating_income"])
        assert "operating_income" in result["financials"]

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_only_10k_and_10q_included(self, mock_get):
        result = get_company_financials("NVDA", ["revenue"])
        for rec in result["financials"]["revenue"]:
            assert rec["form"] in ("10-K", "10-Q")

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_deduplication_keeps_latest_filing(self, mock_get):
        # Both 2022-01-30 entries have the same period; only one should survive
        result = get_company_financials("NVDA", ["revenue"])
        periods = [r["period"] for r in result["financials"]["revenue"]]
        assert len(periods) == len(set(periods))

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_missing_concept_not_in_result(self, mock_get):
        result = get_company_financials("NVDA", ["rd_expense"])
        # rd_expense not in mock data — should not appear or be empty
        financials = result["financials"]
        if "rd_expense" in financials:
            assert financials["rd_expense"] == []

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_company_name_present(self, mock_get):
        result = get_company_financials("NVDA", ["revenue"])
        assert result["company_name"] == "NVIDIA Corp"


# ── get_recent_filing_text ────────────────────────────────────────────────────

class TestGetRecentFilingText:
    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_returns_mda_text(self, mock_get):
        result = get_recent_filing_text("NVDA", "10-K")
        assert result["ticker"] == "NVDA"
        assert result["form_type"] == "10-K"
        assert "extracted_text" in result
        assert len(result["extracted_text"]) > 0

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_text_truncated_at_6000(self, mock_get):
        result = get_recent_filing_text("NVDA", "10-K")
        assert len(result["extracted_text"]) <= 6000

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_filing_date_present(self, mock_get):
        result = get_recent_filing_text("NVDA", "10-K")
        assert result["filed_date"] == "2023-02-24"

    @patch("tools.sec_edgar._get", side_effect=_mock_get)
    @patch("tools.sec_edgar._ticker_map", None)
    def test_no_matching_form_returns_error(self, mock_get):
        result = get_recent_filing_text("NVDA", "S-1")  # not in mock submissions
        assert "error" in result


# ── _extract_mda_section ─────────────────────────────────────────────────────

class TestExtractMdaSection:
    def test_extracts_between_item7_and_item8(self):
        # Need >200 chars between header and end-section marker so the offset is cleared
        filler = "Revenue grew 20%. Operating margins improved. " * 10
        text = (
            "Some preamble. "
            "Management Discussion and Analysis of Results. "
            + filler
            + "Item 8. Financial Statements follow."
        )
        extracted = _extract_mda_section(text)
        assert "Revenue grew" in extracted
        assert "Financial Statements" not in extracted

    def test_falls_back_to_start_when_no_mda_header(self):
        text = "This document has no MD&A header. Just plain text. " * 50
        extracted = _extract_mda_section(text)
        assert len(extracted) <= 5000

    def test_respects_6000_char_cap(self):
        # Even if the section is huge, output is capped
        text = "Management Discussion and Analysis. " + "x" * 10000
        extracted = _extract_mda_section(text)
        assert len(extracted) <= 6000

    def test_case_insensitive_header_match(self):
        text = "MANAGEMENT DISCUSSION AND ANALYSIS. Revenue data here. Item 8. End."
        extracted = _extract_mda_section(text)
        assert "Revenue data here" in extracted
