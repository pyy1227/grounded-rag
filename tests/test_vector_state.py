"""向量状态可见性测试：配置开了向量 ≠ 库里真的有向量。

这条静默降级以前没有任何信号：`vector` 分数一律 0、`reasons` 里没有 vector#，
但接口照常返回。现在要求至少三处能看出来：Retriever.vector_active、日志告警、/health。
"""
from __future__ import annotations

import logging

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
