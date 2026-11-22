"""测试夹具：临时工作目录 + 已入库的迷你知识库。

注意：这里不用 pytest 的 tmp_path——在部分受限环境里系统临时目录不可写，
统一使用项目内的 .tmp-tests/（已加入 .gitignore）。
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from src.ragagent.config import AppConfig
from src.ragagent.indexer import Indexer
from src.ragagent.storage import Storage

DOC = """# 售后政策

## 退货运费

| 退货原因 | 运费承担方 |
| --- | --- |
| 商品质量问题 | 商家承担 |
| 7 天无理由 | 买家承担 |

## 退款时效

退款一般在商家同意后 1-3 个工作日到账。

## 常见问题

Q：退货运费谁承担？
A：质量问题由商家承担，7 天无理由由买家承担。

Q：退款多久到账？
A：原路退回 1-3 个工作日到账。
"""


@pytest.fixture()
def workdir():
    path = Path(".tmp-tests") / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def temp_docs(workdir) -> Path:
    docs = workdir / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "售后政策.md").write_text(DOC, encoding="utf-8")
    return docs


@pytest.fixture()
def temp_env(temp_docs):
    config = AppConfig()
    config.storage.path = str(temp_docs.parent / "rag.db")
    config.retrieval.enable_vector = False  # 单测固定，避免向量噪声
    config.retrieval.enable_rerank = True

    storage = Storage(config.storage.path)
    Indexer(storage, config).ingest_path(temp_docs)
    return config, storage
