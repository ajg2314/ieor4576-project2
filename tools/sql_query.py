"""Tool: DuckDB SQL query execution for the Collector agent.

The agent dynamically writes SQL queries against local CSV/Parquet datasets.
DuckDB lets us query large files without loading them entirely into memory or context.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def execute_sql(query: str, limit: int = 500) -> dict:
    """
    Execute a SQL query using DuckDB against files in the data/ directory.

    The agent writes the SQL at runtime. DuckDB can query CSV/Parquet/JSON
    files directly using paths like: SELECT * FROM 'data/filename.csv'

    Args:
        query: SQL query string (agent-generated at runtime)
        limit: Safety cap on rows returned to agent context (default 500)

    Returns:
        dict with 'columns', 'rows', 'row_count', and 'truncated' fields.
    """
    safe_query = _apply_row_limit(query, limit)

    con = duckdb.connect(database=":memory:")
    # Register data directory so agent can reference files by short name
    con.execute(f"SET search_path TO '{DATA_DIR}'")

    result = con.execute(safe_query).fetchdf()
    con.close()

    return {
        "columns": list(result.columns),
        "rows": result.to_dict(orient="records"),
        "row_count": len(result),
        "truncated": len(result) >= limit,
    }


def _apply_row_limit(query: str, limit: int) -> str:
    """Append LIMIT clause if not already present."""
    normalized = query.strip().upper()
    if "LIMIT" not in normalized:
        return f"{query.rstrip(';')} LIMIT {limit}"
    return query


def list_available_datasets() -> list[str]:
    """Return names of queryable data files in the data/ directory."""
    if not DATA_DIR.exists():
        return []
    return [f.name for f in DATA_DIR.iterdir() if f.suffix in {".csv", ".parquet", ".json"}]
