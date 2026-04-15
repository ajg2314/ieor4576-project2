"""Tests for the TF-IDF RAG store."""

import shutil
import tempfile
from pathlib import Path
import pytest


def test_chunk_text():
    from tools.rag_store import _chunk_text
    text = "Hello world. " * 200  # 2600 chars
    chunks = _chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) >= 100


def test_parse_frontmatter():
    from tools.rag_store import _parse_frontmatter
    text = "---\nsector: semiconductors\ntype: knowledge\n---\n\nBody text here."
    meta, body = _parse_frontmatter(text)
    assert meta["sector"] == "semiconductors"
    assert meta["type"] == "knowledge"
    assert "Body text here" in body


def test_build_and_query_index():
    from tools.rag_store import _build_index, _query_index
    docs = [
        "NVIDIA semiconductor AI chips GPU revenue growth 2024 data center",
        "Intel CPU processor personal computer market share decline",
        "US China export controls semiconductor geopolitical trade war",
        "TSMC foundry Taiwan manufacturing advanced nodes EUV lithography",
    ]
    metas = [{"sector": "semiconductors", "i": str(i)} for i in range(4)]
    index = _build_index(docs, metas)

    # Query about AI chips should return NVIDIA first
    results = _query_index(index, "AI chips GPU data center revenue", n=2)
    assert len(results) >= 1
    assert "NVIDIA" in results[0][1]

    # Query about geopolitics should return export controls doc
    results = _query_index(index, "China export controls trade", n=2)
    assert any("export controls" in doc for _, doc, _ in results)


def test_seed_and_retrieve(tmp_path):
    """Full seed + retrieve cycle using real data files."""
    from tools import rag_store as rs

    # Patch store paths to use temp dir
    original_store = rs.RAG_STORE_DIR
    original_report_idx = rs._REPORT_INDEX
    original_knowledge_idx = rs._KNOWLEDGE_INDEX

    tmp_store = tmp_path / "rag_store"
    rs.RAG_STORE_DIR = tmp_store
    rs._REPORT_INDEX = tmp_store / "report_examples.json"
    rs._KNOWLEDGE_INDEX = tmp_store / "sector_knowledge.json"
    rs._report_index_cache = None
    rs._knowledge_index_cache = None

    try:
        counts = rs.seed_all(force=True)
        assert counts["report_examples"] > 0, "Should have seeded report examples"
        assert counts["sector_knowledge"] > 0, "Should have seeded sector knowledge"

        # Retrieve report example
        ex = rs.retrieve_report_example("semiconductors", "NVIDIA revenue growth AI chips")
        assert len(ex) > 100, "Should return substantial content"
        assert any(word in ex.lower() for word in ["nvidia", "revenue", "semiconductor"])

        # Retrieve sector knowledge
        kb = rs.retrieve_sector_knowledge("semiconductors", "US China export controls CHIPS Act")
        assert len(kb) > 100
        assert any(word in kb.lower() for word in ["export", "china", "chips", "control"])

        # Retrieve EDA playbook
        pb = rs.retrieve_eda_playbook("semiconductors", "who is leading the sector")
        assert len(pb) > 50

    finally:
        # Restore original paths
        rs.RAG_STORE_DIR = original_store
        rs._REPORT_INDEX = original_report_idx
        rs._KNOWLEDGE_INDEX = original_knowledge_idx
        rs._report_index_cache = None
        rs._knowledge_index_cache = None


def test_idempotent_seeding(tmp_path):
    """Seeding twice should not add duplicate chunks (returns 0 on second call)."""
    from tools import rag_store as rs

    original_store = rs.RAG_STORE_DIR
    original_report_idx = rs._REPORT_INDEX
    original_knowledge_idx = rs._KNOWLEDGE_INDEX

    tmp_store = tmp_path / "rag_store2"
    rs.RAG_STORE_DIR = tmp_store
    rs._REPORT_INDEX = tmp_store / "report_examples.json"
    rs._KNOWLEDGE_INDEX = tmp_store / "sector_knowledge.json"
    rs._report_index_cache = None
    rs._knowledge_index_cache = None

    try:
        first = rs.seed_all(force=True)
        second = rs.seed_all()  # force=False — should skip
        assert second["report_examples"] == 0
        assert second["sector_knowledge"] == 0
    finally:
        rs.RAG_STORE_DIR = original_store
        rs._REPORT_INDEX = original_report_idx
        rs._KNOWLEDGE_INDEX = original_knowledge_idx
        rs._report_index_cache = None
        rs._knowledge_index_cache = None
