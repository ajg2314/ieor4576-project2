"""RAG (Retrieval-Augmented Generation) store for the Sector Analyst system.

Two collections:
  1. report_examples  — exemplary analyst memos (style & structure reference for Hypothesis agent)
  2. sector_knowledge — domain knowledge: sector terms, EDA playbooks, geopolitical context

Uses a pure-stdlib TF-IDF implementation — no neural model, no scipy, no sklearn, no API key.
Index is persisted as JSON to data/rag_store/. For a small static corpus (~50-100 chunks)
of keyword-rich technical/financial text, stdlib TF-IDF retrieval is sufficient.

Seeding is idempotent — safe to call on every pipeline startup.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RAG_STORE_DIR = DATA_DIR / "rag_store"
REPORT_EXAMPLES_DIR = DATA_DIR / "report_examples"
SECTOR_KNOWLEDGE_DIR = DATA_DIR / "sector_knowledge"

# Index files (JSON, no pickle — no scipy dependency)
_REPORT_INDEX = RAG_STORE_DIR / "report_examples.json"
_KNOWLEDGE_INDEX = RAG_STORE_DIR / "sector_knowledge.json"

# Common English stopwords
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "as", "if", "not", "no", "so", "up", "out", "about", "into", "than",
    "then", "also", "their", "they", "we", "our", "you", "your", "he", "she",
    "his", "her", "more", "most", "other", "which", "who", "what", "how",
    "when", "where", "can", "all", "each", "over", "after", "while", "both",
}


# ---------------------------------------------------------------------------
# Pure-stdlib TF-IDF index
# Index structure (JSON-serialisable):
#   {"docs": [...str], "metas": [...dict], "idf": {term: float}, "tf": [[{term: float}]]}
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _build_index(docs: list[str], metas: list[dict]) -> dict:
    """Build a TF-IDF index over a list of documents (pure stdlib)."""
    n = len(docs)
    tf_list: list[dict[str, float]] = []
    df: Counter = Counter()

    for doc in docs:
        tokens = _tokenise(doc)
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        tf = {t: c / total for t, c in counts.items()}
        tf_list.append(tf)
        df.update(set(counts.keys()))

    # IDF with smoothing
    idf = {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

    return {"docs": docs, "metas": metas, "tf": tf_list, "idf": idf}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_vec(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}


def _query_index(index: dict, query: str, n: int) -> list[tuple[float, str, dict]]:
    """Query a TF-IDF index. Returns (score, doc, meta) sorted by score desc."""
    q_tokens = _tokenise(query)
    q_counts = Counter(q_tokens)
    q_total = max(len(q_tokens), 1)
    q_tf = {t: c / q_total for t, c in q_counts.items()}
    q_vec = _tfidf_vec(q_tf, index["idf"])

    scores: list[tuple[float, int]] = []
    for i, tf in enumerate(index["tf"]):
        doc_vec = _tfidf_vec(tf, index["idf"])
        scores.append((_cosine(q_vec, doc_vec), i))

    scores.sort(reverse=True)
    return [
        (score, index["docs"][i], index["metas"][i])
        for score, i in scores[:n]
        if score > 0.0
    ]


def _load_index(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load RAG index from %s: %s", path, e)
        return None


def _save_index(index: dict, path: Path) -> None:
    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for better retrieval granularity."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at a paragraph boundary
            break_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if break_pos != -1:
                end = break_pos
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        # Always advance past current start to prevent infinite loop
        next_start = end - overlap
        if next_start <= start:
            next_start = end  # no overlap if it would regress
        start = next_start
        if end == len(text):
            break
    return chunks


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-like frontmatter from a markdown file."""
    metadata: dict[str, str] = {}
    if text.startswith("---"):
        try:
            end_fm = text.index("---", 3)
            for line in text[3:end_fm].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip()
            text = text[end_fm + 3:].strip()
        except ValueError:
            pass
    return metadata, text


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_report_examples(force: bool = False) -> int:
    """Load report examples from data/report_examples/ into the TF-IDF index.

    Idempotent — skips rebuild if index file already exists unless force=True.
    Returns number of chunks indexed.
    """
    if _REPORT_INDEX.exists() and not force:
        return 0  # Already seeded

    docs: list[str] = []
    metas: list[dict] = []

    for md_file in sorted(REPORT_EXAMPLES_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        base_meta = {"filename": md_file.name, "type": "report_example", **fm}

        for i, chunk in enumerate(_chunk_text(body)):
            docs.append(chunk)
            metas.append({**base_meta, "chunk_index": str(i)})

    if not docs:
        return 0

    index = _build_index(docs, metas)
    _save_index(index, _REPORT_INDEX)
    logger.info("RAG report_examples: indexed %d chunks from %d files",
                len(docs), len(list(REPORT_EXAMPLES_DIR.glob("*.md"))))
    return len(docs)


def seed_sector_knowledge(force: bool = False) -> int:
    """Load sector knowledge from data/sector_knowledge/ into the TF-IDF index.

    Idempotent unless force=True. Returns number of chunks indexed.
    """
    if _KNOWLEDGE_INDEX.exists() and not force:
        return 0  # Already seeded

    docs: list[str] = []
    metas: list[dict] = []

    for md_file in sorted(SECTOR_KNOWLEDGE_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        base_meta = {"filename": md_file.name, "sector": fm.get("sector", "general"),
                     "type": fm.get("type", "knowledge"), **fm}

        for i, chunk in enumerate(_chunk_text(body)):
            docs.append(chunk)
            metas.append({**base_meta, "chunk_index": str(i)})

    if not docs:
        return 0

    index = _build_index(docs, metas)
    _save_index(index, _KNOWLEDGE_INDEX)
    logger.info("RAG sector_knowledge: indexed %d chunks from %d files",
                len(docs), len(list(SECTOR_KNOWLEDGE_DIR.glob("*.md"))))
    return len(docs)


def seed_all(force: bool = False) -> dict[str, int]:
    """Seed both indexes. Safe to call on every pipeline startup (idempotent)."""
    try:
        reports_added = seed_report_examples(force=force)
        knowledge_added = seed_sector_knowledge(force=force)
        return {"report_examples": reports_added, "sector_knowledge": knowledge_added}
    except Exception as e:
        logger.warning("RAG seeding failed (non-fatal): %s", e)
        return {"report_examples": 0, "sector_knowledge": 0}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_report_index_cache: dict | None = None
_knowledge_index_cache: dict | None = None


def _get_report_index() -> dict | None:
    global _report_index_cache
    if _report_index_cache is None:
        _report_index_cache = _load_index(_REPORT_INDEX)
    return _report_index_cache


def _get_knowledge_index() -> dict | None:
    global _knowledge_index_cache
    if _knowledge_index_cache is None:
        _knowledge_index_cache = _load_index(_KNOWLEDGE_INDEX)
    return _knowledge_index_cache


def retrieve_report_example(sector: str, question: str, n: int = 2) -> str:
    """Retrieve the most relevant exemplary analyst report sections.

    Args:
        sector: Sector name (e.g. 'semiconductors', 'cloud software')
        question: The user's research question
        n: Number of chunks to retrieve

    Returns:
        Concatenated relevant sections from exemplary reports.
    """
    try:
        index = _get_report_index()
        if index is None:
            seed_report_examples()
            index = _get_report_index()
        if index is None:
            return ""
        query = f"{sector} sector analyst report: {question}"
        results = _query_index(index, query, n=n)
        if not results:
            return ""
        return "\n\n---\n\n".join(doc for _, doc, _ in results)
    except Exception as e:
        logger.warning("RAG retrieve_report_example failed: %s", e)
        return ""


def retrieve_sector_knowledge(sector: str, query: str, n: int = 4) -> str:
    """Retrieve relevant sector domain knowledge.

    Args:
        sector: Sector name
        query: Specific topic or question
        n: Number of chunks to retrieve

    Returns:
        Concatenated relevant knowledge chunks with source labels.
    """
    try:
        index = _get_knowledge_index()
        if index is None:
            seed_sector_knowledge()
            index = _get_knowledge_index()
        if index is None:
            return ""
        full_query = f"{sector}: {query}"
        results = _query_index(index, full_query, n=n)
        if not results:
            return ""
        parts = []
        for _, doc, meta in results:
            source = meta.get("filename", "knowledge base")
            parts.append(f"[Source: {source}]\n{doc}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.warning("RAG retrieve_sector_knowledge failed: %s", e)
        return ""


def retrieve_eda_playbook(sector: str, question: str) -> str:
    """Retrieve EDA analysis guidance for a specific sector and question type."""
    return retrieve_sector_knowledge(sector, f"EDA analysis guidance: {question}", n=3)


# ---------------------------------------------------------------------------
# Web seeding — fetch real reports/articles and add to report_examples
# ---------------------------------------------------------------------------

_WEB_SEED_SOURCES = [
    ("semiconductors", "semiconductor industry outlook AI chips 2024 2025 analyst report"),
    ("semiconductors", "NVIDIA AMD Intel semiconductor market share revenue analysis"),
    ("semiconductors", "US China chip export controls impact semiconductor companies"),
    ("cloud software", "cloud computing market growth AWS Azure Google analyst 2024 2025"),
    ("cloud software", "AI SaaS enterprise software revenue growth outlook"),
    ("energy", "energy sector oil gas renewable analyst outlook 2024 2025"),
    ("automotive", "electric vehicle market growth BYD Tesla analyst report 2024"),
    ("general", "sector geopolitical risk supply chain trade war 2024 2025"),
]


def seed_from_web(max_per_query: int = 2, max_chars_per_page: int = 3000,
                  force: bool = False) -> int:
    """Fetch real analyst articles from the web and add them to the report index.

    Appends to existing index. Returns number of new documents added.
    """
    web_cache_file = RAG_STORE_DIR / "web_docs.json"
    if web_cache_file.exists() and not force:
        existing = json.loads(web_cache_file.read_text())
    else:
        existing = {"urls": [], "docs": [], "metas": []}

    seen_urls = set(existing["urls"])

    try:
        from tools.web_search import search_web, fetch_page_text
    except ImportError:
        logger.warning("web_search not available — skipping web seed")
        return 0

    new_docs: list[str] = []
    new_metas: list[dict] = []

    for sector, query in _WEB_SEED_SOURCES:
        try:
            results = search_web(query, max_results=max_per_query + 3)
            fetched = 0
            for r in results:
                if fetched >= max_per_query:
                    break
                url = r.get("url", "")
                if not url or not url.startswith("http") or url in seen_urls:
                    continue
                page = fetch_page_text(url, max_chars=max_chars_per_page)
                text = page.get("text", "").strip()
                if len(text) < 200:
                    continue
                title = page.get("title", r.get("title", ""))
                new_docs.append(text)
                new_metas.append({
                    "type": "web_report",
                    "sector": sector,
                    "url": url,
                    "title": title[:200],
                    "filename": "web:" + url[:80],
                })
                seen_urls.add(url)
                fetched += 1
        except Exception as e:
            logger.warning("Web seed failed for query '%s': %s", query, e)

    if not new_docs:
        return 0

    # Merge with existing and rebuild the report index
    all_docs = existing["docs"] + new_docs
    all_metas = existing["metas"] + new_metas

    # Save web cache
    web_cache_file.parent.mkdir(parents=True, exist_ok=True)
    web_cache_file.write_text(json.dumps({
        "urls": list(seen_urls),
        "docs": all_docs,
        "metas": all_metas,
    }))

    # Rebuild full report index (local files + web)
    index = _get_report_index()
    local_docs = index["docs"] if index else []
    local_metas = index["metas"] if index else []
    combined_docs = local_docs + all_docs
    combined_metas = local_metas + all_metas
    new_index = _build_index(combined_docs, combined_metas)
    _save_index(new_index, _REPORT_INDEX)

    global _report_index_cache
    _report_index_cache = new_index

    logger.info("RAG web seed: %d new documents added", len(new_docs))
    return len(new_docs)


def seed_all_with_web(max_per_query: int = 2) -> dict[str, int]:
    """Seed local files + web content."""
    counts = seed_all()
    try:
        counts["web_reports"] = seed_from_web(max_per_query=max_per_query)
    except Exception as e:
        logger.warning("Web seed failed (non-fatal): %s", e)
        counts["web_reports"] = 0
    return counts
