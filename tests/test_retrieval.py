"""检索编排测试：命中、拒答、增量跳过。"""
from __future__ import annotations

from ragagent.indexer import Indexer
from ragagent.retrieval.retriever import Retriever


def test_search_hits_expected_document(temp_env):
    config, storage = temp_env
    hits = Retriever(storage, config).search("退货运费谁承担")
    assert hits
    assert hits[0].chunk.doc_id == "售后政策"
    assert "运费" in hits[0].chunk.text


def test_unseen_terms_trigger_refusal(temp_env):
    config, storage = temp_env
    retriever = Retriever(storage, config)
    config.retrieval.max_unseen_ratio = 1.0
    assert retriever.search("量子计算机怎么维修") == []
    assert retriever.last_sources.get("refused_by") in {"all_terms_unseen", "unseen_ratio"}


def test_incremental_ingest_skips_unchanged(temp_env, temp_docs):
    config, storage = temp_env
    report = Indexer(storage, config).ingest_path(temp_docs)
    assert report["skipped"], "文件未变化时应跳过"
    assert not report["added"] and not report["updated"]


def test_force_ingest_rebuilds(temp_env, temp_docs):
    config, storage = temp_env
    report = Indexer(storage, config).ingest_path(temp_docs, force=True)
    assert report["updated"]


def test_parent_expansion_returns_section_text(temp_env):
    config, storage = temp_env
    parent = [c for c in storage.iter_chunks(include_parents=True) if c.is_parent]
    if not parent:
        return  # demo 文档可能没有长章节，跳过
    retriever = Retriever(storage, config)
    hits = retriever.search("售后", top_k=5)
    child = next((h for h in hits if h.chunk.parent_id), None)
    if child is None:
        return
    assert len(retriever.expand(child)) >= len(child.chunk.text)


def test_ingest_records_bulk_for_external_corpus(temp_env):
    """外部评测集走「一条记录 = 一个 chunk」的旁路：不解析、不切分。"""
    config, storage = temp_env
    indexer = Indexer(storage, config)
    records = [
        {"doc_id": "p0", "title": "passage", "text": "充电时间约 2.5 小时，支持 5V/2A 输入。"},
        {"doc_id": "p1", "title": "passage", "text": "风扇有三档风力，最大风速 4.2 m/s。"},
    ]
    report = indexer.ingest_records(records)

    assert report["chunks"] == 2
    doc = storage.get_document("p0")
    assert doc is not None and doc.chunk_count == 1
    assert storage.count_chunks("p0") == 1
    assert storage.count_embeddings(indexer.embedder.model) >= 2

    hits = Retriever(storage, config).search("充电要多久", top_k=3)
    assert hits and hits[0].chunk.doc_id == "p0"
