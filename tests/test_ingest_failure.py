"""入库失败可见性：CLI 必须返回非零，统计不能把失败文件算成 documents。"""
from __future__ import annotations

from pathlib import Path

from ragagent.cli import main
from ragagent.config import AppConfig
from ragagent.indexer import Indexer
from ragagent.storage import Storage

DOC = "# 售后政策\n\n## 退货运费\n\n质量问题由商家承担。\n"


def _setup(workdir: Path) -> tuple[Path, Path]:
    docs = workdir / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "售后政策.md").write_text(DOC, encoding="utf-8")
    config_path = workdir / "config.toml"
    config_path.write_text(
        f'[storage]\npath = "{(workdir / "rag.db").as_posix()}"\n',
        encoding="utf-8",
    )
    return docs, config_path


def _fail_embedding(self, texts):
    raise RuntimeError("embedding service down")


def test_ingest_report_does_not_count_failed_files(workdir, monkeypatch):
    docs, _ = _setup(workdir)
    config = AppConfig()
    config.storage.path = str(workdir / "rag.db")
    storage = Storage(config.storage.path)
    monkeypatch.setattr(Indexer, "_embed", _fail_embedding)

    report = Indexer(storage, config).ingest_path(docs)

    assert report["chunks"] == 0
    assert len(report["errors"]) == 1
    assert report["stats"]["documents"] == 0
    assert report["stats"]["attempted"] == 1
    assert report["stats"]["errors"] == 1
    assert storage.count_chunks() == 0


def test_cli_ingest_returns_nonzero_when_embedding_fails(workdir, monkeypatch):
    docs, config_path = _setup(workdir)
    monkeypatch.setattr(Indexer, "_embed", _fail_embedding)

    code = main(["--config", str(config_path), "ingest", str(docs)])

    assert code == 1
