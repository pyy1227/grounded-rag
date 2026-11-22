"""BM25 与融合策略测试。"""
from __future__ import annotations

import pytest

from ragagent.models import Chunk
from ragagent.retrieval.bm25 import BM25Index
from ragagent.retrieval.fusion import rrf_fuse, weighted_fuse


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        ordinal=0,
        section_path="章节",
        kind="prose",
        text=text,
    )


def test_bm25_prefers_matching_chunk():
    chunks = [
        _chunk("a", "退货运费由商家承担，质量问题包运费。"),
        _chunk("b", "发票在确认收货后三个工作日开出。"),
        _chunk("c", "风扇续航八小时，支持三档风力。"),
    ]
    hits = BM25Index(chunks).search("退货运费", top_k=3)
    assert hits
    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].score > 0


def test_bm25_returns_empty_for_unknown_terms():
    chunks = [_chunk("a", "退货运费由商家承担。")]
    assert BM25Index(chunks).search("量子计算机") == []


def test_rrf_prefers_chunk_ranked_high_by_both_sources():
    fused = rrf_fuse({"bm25": ["a", "b", "c"], "vector": ["a", "c", "b"]})
    assert max(fused, key=fused.get) == "a"
    assert fused["a"] == pytest.approx(1.0)
    assert set(fused) == {"a", "b", "c"}


def test_weighted_fusion_respects_weights():
    scores = {"bm25": {"a": 1.0, "b": 0.2}, "vector": {"b": 1.0}}
    bm25_heavy = weighted_fuse(scores, {"bm25": 0.9, "vector": 0.1})
    vector_heavy = weighted_fuse(scores, {"bm25": 0.1, "vector": 0.9})
    assert max(bm25_heavy, key=bm25_heavy.get) == "a"
    assert max(vector_heavy, key=vector_heavy.get) == "b"


def test_fusion_handles_empty_sources():
    assert rrf_fuse({}) == {}
    assert weighted_fuse({}, {"bm25": 1.0}) == {}
