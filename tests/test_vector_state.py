"""向量状态可见性测试：配置开了向量，不等于库里真有向量。

以前静默降级时没有任何信号：vector 分数一律 0、reasons 里没有 vector#，接口照常返回。
现在要求三处能看出来：Retriever.vector_active、日志告警、/health。
"""
from __future__ import annotations

import logging

import numpy as np
from fastapi.testclient import TestClient

import ragagent.embedding as embedding
from ragagent.api import create_app
from ragagent.embedding import HashEmbedder
from ragagent.retrieval.retriever import Retriever
from ragagent.tokenization import mode as tokenizer_mode
from ragagent.tokenization import tokenize


def _drop_embeddings(storage) -> None:
    with storage.conn() as conn:
        conn.execute("DELETE FROM embeddings")


def test_hash_embedder_uses_the_shared_tokenizer():
    """哈希向量的分词必须和 BM25 走同一套，否则 model 名会撒谎。"""
    assert embedding._tokenize is tokenize
    assert embedding._tokenize("退货运费谁承担") == tokenize("退货运费谁承担")
    assert HashEmbedder(8).model == f"hash-8-{tokenizer_mode()}"


def test_missing_vectors_warns_instead_of_degrading_silently(temp_env, caplog):
    config, storage = temp_env
    _drop_embeddings(storage)
    config.retrieval.enable_vector = True

    retriever = Retriever(storage, config)
    with caplog.at_level(logging.WARNING):
        hits = retriever.search("退货运费谁承担")

    assert hits, "向量缺失不应影响 BM25 主路径"
    assert retriever.vector_active is False
    assert retriever.vector_count == 0
    assert any("没有 model=" in record.getMessage() for record in caplog.records)


def test_health_exposes_real_vector_state(temp_env):
    config, storage = temp_env
    body = TestClient(create_app(config=config)).get("/health").json()

    assert body["vector_enabled"] is False
    assert body["vectors"] == storage.count_embeddings(body["embedder"])
    assert body["vectors"] > 0, "入库时总会写向量，即使检索侧关掉了向量"
    assert body["vector_models_in_db"]


def test_transient_embedding_failure_retries_instead_of_permanent_degrade(temp_env):
    """一次 embedding 失败只应进入冷却，不能把向量索引永久置空。"""
    config, storage = temp_env
    config.retrieval.enable_vector = True
    model = "flaky"
    chunks = storage.iter_chunks()
    storage.save_embeddings(
        model,
        {chunk.chunk_id: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for chunk in chunks},
    )

    class FlakyEmbedder:
        def __init__(self) -> None:
            self.model = model
            self.dim = 4
            self.fail = True

        def embed(self, texts):
            if self.fail:
                raise RuntimeError("embedding down")
            return np.ones((len(texts), 4), dtype=np.float32)

    embedder = FlakyEmbedder()
    retriever = Retriever(storage, config, embedder=embedder)

    hits = retriever.search("退货运费谁承担")
    assert hits, "向量失败不能影响 BM25 主路径"
    assert retriever.vector_active is False
    assert retriever.index_status()["retry_pending"] is True

    embedder.fail = False
    retriever._vector_retry_at = 0.0
    retriever.search("退货运费谁承担")
    assert retriever.vector_active is True, "冷却结束后应自动恢复向量检索"


def test_health_reflects_runtime_vector_failure(temp_env):
    """/health 报运行态：embedding 失败后 vector_active=false，但库里仍有向量。"""
    config, storage = temp_env
    config.retrieval.enable_vector = True
    app = create_app(config=config)
    retriever = app.state.retriever

    class FailingEmbedder:
        model = retriever.embedder.model

        def embed(self, texts):
            raise RuntimeError("embedding down")

    retriever.embedder = FailingEmbedder()
    client = TestClient(app)
    client.post("/search", json={"query": "退货运费谁承担"})

    body = client.get("/health").json()
    assert body["vector_active"] is False
    assert body["vectors"] > 0
    assert body["vector_retry_pending"] is True
