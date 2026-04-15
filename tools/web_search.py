"""Web search and page-fetch tools for the Research agent.

Uses DuckDuckGo (no API key required) for search and httpx for page fetching.
Results are text-extracted and truncated to stay within LLM context limits.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

HEADERS = {
    "User-Agent": "IEOR4576 Research Agent research@columbia.edu",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_last_request = 0.0
_MIN_INTERVAL = 0.5  # be polite to public services


def _get(url: str, params: dict | None = None, timeout: float = 20.0) -> httpx.Response:
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    resp = httpx.get(url, params=params, headers=HEADERS,
                     timeout=timeout, follow_redirects=True)
    _last_request = time.monotonic()
    return resp


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def search_web(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Search DuckDuckGo and return a list of results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 8).

    Returns:
        List of dicts with 'title', 'url', 'snippet' keys.
    """
    try:
        # DuckDuckGo HTML search (lite endpoint — simpler to parse)
        resp = _get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
        )
        html = resp.text

        # Extract result blocks
        results: list[dict[str, str]] = []

        # Title + URL from result links
        link_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(links[:max_results]):
            # DDG redirects: extract actual URL
            real_url = url
            uddg_match = re.search(r"uddg=([^&]+)", url)
            if uddg_match:
                import urllib.parse
                real_url = urllib.parse.unquote(uddg_match.group(1))

            snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
            results.append({
                "title": _strip_html(title).strip(),
                "url": real_url,
                "snippet": snippet[:300],
            })

        return results

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("search_web failed for '%s': %s", query, e)
        return []


def fetch_page_text(url: str, max_chars: int = 4000) -> dict[str, Any]:
    """Fetch a web page and return its main text content.

    Strips HTML, navigation, and boilerplate. Good for reading analyst
    reports, news articles, and research papers.

    Args:
        url: Full URL to fetch.
        max_chars: Maximum characters to return (default 4000).

    Returns:
        Dict with 'url', 'title', 'text', 'char_count' keys.
    """
    try:
        resp = _get(url, timeout=25.0)
        html = resp.text

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = _strip_html(title_match.group(1)) if title_match else ""

        # Try to find the main content block (article / main / body)
        for tag in ("article", "main", r'div[^>]*class="[^"]*(?:content|article|body|post)[^"]*"'):
            pattern = re.compile(rf"<{tag}[^>]*>(.*?)</{tag.split('[')[0]}>",
                                  re.DOTALL | re.IGNORECASE)
            match = pattern.search(html)
            if match:
                html = match.group(1)
                break

        text = _strip_html(html)

        # Remove very short lines (nav links, buttons)
        lines = [ln.strip() for ln in text.split("  ") if len(ln.strip()) > 40]
        text = " ".join(lines)

        return {
            "url": url,
            "title": title[:200],
            "text": text[:max_chars],
            "char_count": len(text),
        }

    except Exception as e:
        return {
            "url": url,
            "title": "",
            "text": "",
            "char_count": 0,
            "error": str(e),
        }
