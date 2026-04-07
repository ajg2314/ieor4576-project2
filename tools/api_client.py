"""Tool: External REST API integration for the Collector agent.

The agent calls this tool with a URL and optional query parameters constructed
at runtime based on the user's question. No data or endpoints are hard-coded
into the system prompt.
"""

import os
import httpx
from typing import Any


async def fetch_api(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """
    Fetch data from an external REST API endpoint.

    Args:
        url: Full API endpoint URL (constructed dynamically by the agent)
        params: Query parameters dict
        headers: Optional request headers (auth tokens injected server-side)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as a dict, with metadata fields added.
    """
    api_key = os.environ.get("DATA_API_KEY", "")
    default_headers: dict[str, str] = {}
    if api_key:
        default_headers["Authorization"] = f"Bearer {api_key}"
    if headers:
        default_headers.update(headers)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params, headers=default_headers)
        response.raise_for_status()
        data = response.json()

    return {
        "status_code": response.status_code,
        "url": str(response.url),
        "data": data,
    }
