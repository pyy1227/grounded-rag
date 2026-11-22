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
    config.retrieval.max_unseen_ratio = 0.8
    retriever = Retriever(storage, config)
    assert retriever.search("量子计算机怎么维修") == []
    assert retriever.last_sources.get("refused_by") == "unseen_ratio"


def test_content_tokens_drop_particles_and_cross_word_bigrams():
    """拒答信号只看实词：语气助词要滤掉，bigram 不能跨虚词生成。"""
    from ragagent.tokenization import content_tokens

    tokens = content_tokens("能带进机舱吗")
    assert "吗" not in tokens and "能" not in tokens
    assert "机舱" in tokens

    # 「拆都没拆」直接滑窗会切出 拆都/都没；滤掉虚词后只保留相邻的 没拆
    tokens = content_tokens("拆都没拆")
    assert "没拆" in tokens
    assert "拆都" not in tokens


def test_verifier_decides_refusal_when_available(temp_env):
    """有语义校验时以模型判定为准：说不能答就拒，说能答就不拒。"""
    from ragagent.llm import LLMResult
    from ragagent.verify import build_verifier

    config, storage = temp_env
    config.retrieval.max_unseen_ratio = 0.0  # 词面信号一律拒，用来验证模型优先
    retriever = Retriever(storage, config)

    class FakeChat:
        name = "fake"
        verdict = "answerable"

        def chat(self, messages, tools=None, tool_choice="auto", temperature=None, max_tokens=None):
            payload = '{"verdict": "%s", "reason": "test"}' % self.verdict
            return LLMResult(content=payload)

    chat = FakeChat()
    retriever.verifier = build_verifier(chat)

    assert retriever.search("退货运费谁承担"), "模型说能答时不应拒答"

    chat.verdict = "cannot"
    retriever.verifier.cache.clear()
    assert retriever.search("退货运费谁承担") == []
    assert retriever.last_refusal_reason == "verifier_cannot"


def test_need_more_triggers_second_retrieval(temp_env):
    """判定 need_more 时应扩大窗口重试，而不是直接拒答。"""
    config, storage = temp_env
    retriever = Retriever(storage, config)
    retriever._ensure_index()

    calls = {"n": 0}

    class StubVerifier:
        name = "stub"
        cache = {}

        def verify_tiered(self, query, hits):
            calls["n"] += 1
            # 第一次说"还需要资料"，重试后说"能答"——验证二次检索路径
            if calls["n"] == 1:
                return "need_more", "缺关键信息"
            return "answerable", "够了"

        def verify(self, query, hits):
            return self.verify_tiered(query, hits)[0] == "answerable", ""

    retriever.verifier = StubVerifier()
    config.retrieval.top_k = 1  # 第一次只拿 1 条，重试才拿到更宽的窗口
    config.retrieval.verify_retry_top_k = 5
    hits = retriever.search("退货运费谁承担")

    assert hits, "二次检索命中后不应拒答"
    assert calls["n"] >= 2, "应当触发第二次校验"
    assert retriever.last_sources.get("retried") == 1
    assert retriever.last_refusal_reason is None


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
