"""Tools: Statistical aggregation and grouping for the EDA agent.

These are deterministic computation tools — the agent calls them with
specific column names and parameters extracted from the DataBundle.
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from typing import Any


def compute_statistics(
    records: list[dict[str, Any]],
    numeric_columns: list[str],
    group_by: str | None = None,
) -> dict[str, Any]:
    """
    Compute descriptive statistics over retrieved data.

    Args:
        records: List of data records (from DataBundle.records)
        numeric_columns: Column names to compute stats for
        group_by: Optional column to group by before computing stats

    Returns:
        Dict of statistics: mean, median, std, min, max, growth_rate (if time-ordered)
    """
    df = pd.DataFrame(records)

    # Coerce numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if group_by and group_by in df.columns:
        result: dict[str, Any] = {"grouped_by": group_by, "groups": {}}
        for name, group in df.groupby(group_by):
            result["groups"][str(name)] = _stats_for_df(group, numeric_columns)
        return result
    else:
        return _stats_for_df(df, numeric_columns)


def _stats_for_df(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        entry: dict[str, Any] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "count": int(series.count()),
        }
        # Growth rate (first → last, if numeric index is meaningful)
        if len(series) >= 2:
            first, last = float(series.iloc[0]), float(series.iloc[-1])
            if first != 0:
                entry["growth_rate_pct"] = round((last - first) / abs(first) * 100, 2)
        # Correlation with other numeric columns (only columns that exist in df)
        existing_cols = [c for c in numeric_columns if c in df.columns]
        numeric_df = df[existing_cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric_df.corr()
        if col in corr.columns:
            entry["correlations"] = {
                c: round(float(corr[col][c]), 3)
                for c in corr.columns
                if c != col and not np.isnan(corr[col][c])
            }
        stats[col] = entry
    return stats


def group_and_filter(
    records: list[dict[str, Any]],
    filter_column: str,
    filter_value: Any | None = None,
    filter_gt: float | None = None,
    filter_lt: float | None = None,
    group_by: str | None = None,
    aggregate_column: str | None = None,
    aggregate_fn: str = "mean",
) -> dict[str, Any]:
    """
    Filter records by a condition and optionally group/aggregate.

    Args:
        records: Input records
        filter_column: Column to filter on
        filter_value: Exact match value (string or numeric)
        filter_gt: Keep rows where filter_column > this value
        filter_lt: Keep rows where filter_column < this value
        group_by: Column to group results by
        aggregate_column: Column to aggregate after grouping
        aggregate_fn: 'mean', 'sum', 'count', 'max', 'min'

    Returns:
        Filtered (and optionally grouped/aggregated) records
    """
    df = pd.DataFrame(records)

    if filter_value is not None and filter_column in df.columns:
        df = df[df[filter_column] == filter_value]
    if filter_gt is not None and filter_column in df.columns:
        df[filter_column] = pd.to_numeric(df[filter_column], errors="coerce")
        df = df[df[filter_column] > filter_gt]
    if filter_lt is not None and filter_column in df.columns:
        df[filter_column] = pd.to_numeric(df[filter_column], errors="coerce")
        df = df[df[filter_column] < filter_lt]

    if group_by and group_by in df.columns and aggregate_column and aggregate_column in df.columns:
        df[aggregate_column] = pd.to_numeric(df[aggregate_column], errors="coerce")
        fn_map = {"mean": "mean", "sum": "sum", "count": "count", "max": "max", "min": "min"}
        agg_df = df.groupby(group_by)[aggregate_column].agg(fn_map.get(aggregate_fn, "mean")).reset_index()
        return {
            "filter_applied": {"column": filter_column, "value": filter_value, "gt": filter_gt, "lt": filter_lt},
            "grouped_by": group_by,
            "aggregated": aggregate_fn,
            "rows": agg_df.to_dict(orient="records"),
            "row_count": len(agg_df),
        }

    return {
        "filter_applied": {"column": filter_column, "value": filter_value, "gt": filter_gt, "lt": filter_lt},
        "rows": df.head(200).to_dict(orient="records"),
        "row_count": len(df),
    }
