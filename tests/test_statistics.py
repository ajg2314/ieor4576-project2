"""Tests for tools/statistics.py — compute_statistics and group_and_filter."""
import pytest
from tools.statistics import compute_statistics, group_and_filter


RECORDS = [
    {"ticker": "NVDA", "year": "2021", "revenue": 16675, "op_income": 4532},
    {"ticker": "NVDA", "year": "2022", "revenue": 26974, "op_income": 10041},
    {"ticker": "NVDA", "year": "2023", "revenue": 44870, "op_income": 24796},
    {"ticker": "AMD",  "year": "2021", "revenue": 16434, "op_income": 3648},
    {"ticker": "AMD",  "year": "2022", "revenue": 23601, "op_income": 1264},
    {"ticker": "AMD",  "year": "2023", "revenue": 22680, "op_income": 401},
]


class TestComputeStatistics:
    def test_basic_stats_returned(self):
        result = compute_statistics(RECORDS, ["revenue"])
        assert "revenue" in result
        s = result["revenue"]
        assert "mean" in s and "median" in s and "std" in s
        assert "min" in s and "max" in s and "count" in s

    def test_mean_value(self):
        result = compute_statistics(RECORDS, ["revenue"])
        expected_mean = sum(r["revenue"] for r in RECORDS) / len(RECORDS)
        assert abs(result["revenue"]["mean"] - expected_mean) < 1

    def test_min_max(self):
        result = compute_statistics(RECORDS, ["revenue"])
        assert result["revenue"]["min"] == 16434
        assert result["revenue"]["max"] == 44870

    def test_count(self):
        result = compute_statistics(RECORDS, ["revenue"])
        assert result["revenue"]["count"] == 6

    def test_growth_rate_present(self):
        result = compute_statistics(RECORDS, ["revenue"])
        assert "growth_rate_pct" in result["revenue"]

    def test_multiple_columns(self):
        result = compute_statistics(RECORDS, ["revenue", "op_income"])
        assert "revenue" in result
        assert "op_income" in result

    def test_missing_column_skipped(self):
        result = compute_statistics(RECORDS, ["revenue", "nonexistent_col"])
        assert "revenue" in result
        assert "nonexistent_col" not in result

    def test_group_by(self):
        result = compute_statistics(RECORDS, ["revenue"], group_by="ticker")
        assert "grouped_by" in result
        assert "NVDA" in result["groups"]
        assert "AMD" in result["groups"]
        assert result["groups"]["NVDA"]["revenue"]["count"] == 3
        assert result["groups"]["AMD"]["revenue"]["count"] == 3

    def test_correlations_included(self):
        result = compute_statistics(RECORDS, ["revenue", "op_income"])
        assert "correlations" in result["revenue"]
        assert "op_income" in result["revenue"]["correlations"]

    def test_empty_records(self):
        result = compute_statistics([], ["revenue"])
        assert result == {}

    def test_non_numeric_coerced(self):
        records = [{"v": "100"}, {"v": "200"}, {"v": "300"}]
        result = compute_statistics(records, ["v"])
        assert result["v"]["mean"] == 200.0


class TestGroupAndFilter:
    def test_filter_by_exact_value(self):
        result = group_and_filter(RECORDS, "ticker", filter_value="NVDA")
        assert result["row_count"] == 3
        assert all(r["ticker"] == "NVDA" for r in result["rows"])

    def test_filter_gt(self):
        result = group_and_filter(RECORDS, "revenue", filter_gt=20000)
        assert result["row_count"] == 4
        assert all(r["revenue"] > 20000 for r in result["rows"])

    def test_filter_lt(self):
        result = group_and_filter(RECORDS, "revenue", filter_lt=17000)
        assert result["row_count"] == 2

    def test_no_filter_returns_all(self):
        result = group_and_filter(RECORDS, "ticker")
        assert result["row_count"] == len(RECORDS)

    def test_group_and_aggregate_mean(self):
        result = group_and_filter(
            RECORDS, "ticker",
            group_by="ticker", aggregate_column="revenue", aggregate_fn="mean",
        )
        assert "rows" in result
        rows = {r["ticker"]: r["revenue"] for r in result["rows"]}
        expected_nvda = (16675 + 26974 + 44870) / 3
        assert abs(rows["NVDA"] - expected_nvda) < 1

    def test_group_and_aggregate_sum(self):
        result = group_and_filter(
            RECORDS, "ticker",
            group_by="ticker", aggregate_column="revenue", aggregate_fn="sum",
        )
        rows = {r["ticker"]: r["revenue"] for r in result["rows"]}
        assert rows["NVDA"] == 16675 + 26974 + 44870

    def test_group_and_aggregate_max(self):
        result = group_and_filter(
            RECORDS, "ticker",
            group_by="ticker", aggregate_column="op_income", aggregate_fn="max",
        )
        rows = {r["ticker"]: r["op_income"] for r in result["rows"]}
        assert rows["NVDA"] == 24796

    def test_filter_then_group(self):
        result = group_and_filter(
            RECORDS, "revenue", filter_gt=20000,
            group_by="ticker", aggregate_column="revenue", aggregate_fn="mean",
        )
        # Only NVDA 2022/2023 and AMD 2022 qualify (revenue > 20000)
        assert result["row_count"] == 2  # NVDA and AMD (AMD 2023=22680 > 20000 too)

    def test_empty_records(self):
        result = group_and_filter([], "ticker", filter_value="NVDA")
        assert result["row_count"] == 0
