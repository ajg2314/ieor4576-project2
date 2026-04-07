"""Tests for tools/visualizer.py — professional chart generation."""
import os
from pathlib import Path
import pytest
from tools.visualizer import line_chart, bar_chart, waterfall_chart, ARTIFACTS_DIR


SAMPLE_SERIES = {
    "AAPL": {"2021": 365.82, "2022": 394.33, "2023": 383.29},
    "MSFT": {"2021": 168.09, "2022": 198.27, "2023": 211.92},
}

SAMPLE_MARGINS = {
    "AAPL": {"2021": 29.8, "2022": 30.3, "2023": 30.1},
    "MSFT": {"2021": 42.1, "2022": 44.6, "2023": 45.4},
}

WATERFALL_DATA = {
    "Revenue": 383.29,
    "COGS": -223.17,
    "Gross Profit": 160.12,
}


class TestLineChart:
    def test_returns_artifact_path(self):
        path = line_chart(SAMPLE_SERIES, "Revenue Trend", filename="test_line")
        assert path.startswith("artifacts/")
        assert path.endswith(".png")

    def test_file_actually_created(self):
        path = line_chart(SAMPLE_SERIES, "Revenue Trend", filename="test_line2")
        full = ARTIFACTS_DIR / Path(path).name
        assert full.exists()
        assert full.stat().st_size > 0

    def test_billions_format(self):
        path = line_chart(SAMPLE_SERIES, "Revenue", y_format="billions", filename="test_line_b")
        assert path.endswith(".png")

    def test_pct_format(self):
        path = line_chart(SAMPLE_MARGINS, "Operating Margin", y_format="pct", filename="test_line_pct")
        assert path.endswith(".png")

    def test_single_series(self):
        path = line_chart({"AAPL": {"2021": 365.82, "2022": 394.33}}, "AAPL Revenue", filename="test_line_single")
        assert path.endswith(".png")

    def test_subtitle_accepted(self):
        path = line_chart(SAMPLE_SERIES, "Revenue", subtitle="Annual 10-K | USD billions", filename="test_line_sub")
        assert path.endswith(".png")


class TestBarChart:
    def test_returns_artifact_path(self):
        path = bar_chart(SAMPLE_MARGINS, "Operating Margin %", filename="test_bar")
        assert path.startswith("artifacts/")
        assert path.endswith(".png")

    def test_file_actually_created(self):
        path = bar_chart(SAMPLE_MARGINS, "Operating Margin %", filename="test_bar2")
        full = ARTIFACTS_DIR / Path(path).name
        assert full.exists()
        assert full.stat().st_size > 0

    def test_billions_format(self):
        path = bar_chart(SAMPLE_SERIES, "Revenue", y_format="billions", filename="test_bar_b")
        assert path.endswith(".png")

    def test_single_series(self):
        path = bar_chart({"AAPL": {"2021": 29.8, "2022": 30.3}}, "Margin", filename="test_bar_single")
        assert path.endswith(".png")


class TestWaterfallChart:
    def test_returns_artifact_path(self):
        path = waterfall_chart(WATERFALL_DATA, "Revenue Bridge", filename="test_wf")
        assert path.startswith("artifacts/")
        assert path.endswith(".png")

    def test_file_actually_created(self):
        path = waterfall_chart(WATERFALL_DATA, "Revenue Bridge", filename="test_wf2")
        full = ARTIFACTS_DIR / Path(path).name
        assert full.exists()
        assert full.stat().st_size > 0

    def test_pct_format(self):
        pct_data = {"Start": 30.0, "Expansion": 2.5, "End": 32.5}
        path = waterfall_chart(pct_data, "Margin Bridge", y_format="pct", filename="test_wf_pct")
        assert path.endswith(".png")


class TestUniqueFilenames:
    def test_same_filename_different_paths(self):
        """Each call should produce a unique file (uuid suffix)."""
        p1 = line_chart(SAMPLE_SERIES, "T1", filename="uniqueness_test")
        p2 = line_chart(SAMPLE_SERIES, "T2", filename="uniqueness_test")
        assert p1 != p2
