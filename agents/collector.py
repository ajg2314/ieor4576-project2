"""Collector Agent — retrieves real-world data from external sources.

Step 1: Collect. Two retrieval methods:
  1. REST API calls (dynamic endpoint + params constructed at runtime)
  2. SQL queries via DuckDB on local large datasets (CSV/Parquet)
"""

from __future__ import annotations

import os
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from tools.api_client import fetch_api
from tools.sql_query import execute_sql, list_available_datasets
from models.schemas import DataBundle

LITELLM_MODEL_ID = f"vertex_ai/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}"


def _make_model() -> LitellmModel:
    return LitellmModel(
        model=LITELLM_MODEL_ID,
        extra_kwargs={
            "vertex_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "vertex_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        },
    )


COLLECTOR_PROMPT = """\
You are the Data Collector agent in a multi-agent data analysis system.

Your sole responsibility is to retrieve real-world data relevant to the user's
analytics question. You have two tools:

1. fetch_api — call any public REST API. Construct the URL and query parameters
   dynamically based on the question. Do not use hard-coded data.
   Examples: FRED economic data, Open-Meteo weather, NYC Open Data, sports APIs.

2. execute_sql — write and run SQL queries using DuckDB against large local
   datasets (CSV/Parquet). Use list_datasets to see what files are available.

Steps:
1. Decide which data source(s) are most relevant to the question.
2. Call fetch_api and/or execute_sql to retrieve data.
3. Return a structured DataBundle summarizing what you retrieved and why.

Important:
- Do not hard-code data into your response.
- Data must be retrieved at runtime via tool calls.
- If the first retrieval is insufficient, try a refined query.
- Return the raw records plus a short summary of what you found.
"""


@function_tool
async def fetch_api_tool(url: str, params: dict | None = None) -> dict:
    """Fetch data from a REST API endpoint with optional query parameters."""
    return await fetch_api(url, params=params)


@function_tool
def sql_query_tool(query: str) -> dict:
    """Execute a SQL query using DuckDB against local dataset files."""
    return execute_sql(query)


@function_tool
def list_datasets() -> list[str]:
    """List available dataset files that can be queried with SQL."""
    return list_available_datasets()


def build_collector_agent() -> Agent:
    return Agent(
        name="Collector",
        model=_make_model(),
        instructions=COLLECTOR_PROMPT,
        tools=[fetch_api_tool, sql_query_tool, list_datasets],
        output_type=DataBundle,
    )
