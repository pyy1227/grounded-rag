"""入库流水线：解析 → 切分 → 向量化 → 落库（增量）。

幂等设计：
- 文件 sha256 不变且未指定 --force 时跳过，避免重复 embedding 计费；
- 整篇替换 chunk，文档更新不会残留旧块；
- 先算完 chunk/向量再写库，embedding 失败不会破坏已有索引。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .chunking import chunk_document, chunk_stats
from .config import AppConfig
from .embedding import Embedder, build_embedder
from .models import DocumentRecord
from .parsing import SUPPORTED_SUFFIXES, parse_document
from .storage import Storage
from .tokenization import mode as tokenizer_mode


@dataclass
class IngestReport:
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)
    chunks: int = 0
    elapsed_ms: int = 0
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "errors": self.errors,
            "chunks": self.chunks,
            "elapsed_ms": self.elapsed_ms,
            "stats": self.stats,
        }


class Indexer:
    """文档入库服务。"""

    def __init__(
        self,
        storage: Storage,
        config: AppConfig,
        embedder: Optional[Embedder] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.storage = storage
        self.config = config
        self.embedder = embedder or build_embedder(config)
        # 未显式指定时跟随配置：remote embedding 服务通常有单次条数上限
        self.batch_size = max(1, int(batch_size or config.embedding.batch_size))

    def ingest_path(self, path: str | Path, root: Optional[str | Path] = None, force: bool = False) -> dict:
        """单文件/单目录入库，返回文字报告用的统计。"""
        started = time.perf_counter()
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"路径不存在: {target}")

        files: List[Path] = []
        if target.is_dir():
            root_path = Path(root) if root else target
            files = [
                item for item in sorted(target.rglob("*"))
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            ]
        elif target.suffix.lower() in SUPPORTED_SUFFIXES:
            root_path = Path(root) if root else target.parent
            files = [target]
        else:
            raise ValueError(f"不支持的文件类型: {target.suffix}")

        report = IngestReport()
        total_children = 0
        kind_counts: dict = {}

        for file_path in files:
            try:
                status, children, stats = self._ingest_file(file_path, root_path, force)
                report.chunks += children
                total_children += children
                for key, value in stats.items():
                    kind_counts[key] = kind_counts.get(key, 0) + value
                if status == "added":
                    report.added.append(str(file_path))
                elif status == "updated":
                    report.updated.append(str(file_path))
                else:
                    report.skipped.append(str(file_path))
            except Exception as exc:
                report.errors.append({"path": str(file_path), "error": str(exc)})

        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        report.stats = {
            "documents": len(files),
            "children": total_children,
            "kinds": kind_counts,
            "embedder": self.embedder.model,
            "tokenizer": tokenizer_mode(),
        }
        return report.to_dict()

    def _ingest_file(self, path: Path, root: Path, force: bool) -> tuple[str, int, dict]:
        sha = _sha256(path)
        path_str = str(path.resolve())
        existing = self.storage.find_document_by_path(path_str)
        if existing and existing.sha256 == sha and not force:
            return "skipped", 0, {}

        parsed = parse_document(path, root)
        chunks = chunk_document(parsed, self.config.chunking)
        children = [chunk for chunk in chunks if not chunk.is_parent]
        if not children:
            raise ValueError("文档没有可入库的内容")

        vectors = self._embed([chunk.text for chunk in children])
        vector_map = {chunk.chunk_id: vector for chunk, vector in zip(children, vectors)}

        self.storage.replace_chunks(parsed.doc_id, chunks)
        self.storage.save_embeddings(self.embedder.model, vector_map)

        version = (existing.version + 1) if existing else 1
        record = DocumentRecord(
            doc_id=parsed.doc_id,
            title=parsed.title,
            path=path_str,
            sha256=sha,
            size=path.stat().st_size,
            mtime=path.stat().st_mtime,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            version=version,
            chunk_count=len(children),
        )
        self.storage.upsert_document(record)
        self.storage.set_meta("last_ingest_at", record.ingested_at)
        return ("updated" if existing else "added"), len(children), chunk_stats(chunks)

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            matrix = self.embedder.embed(batch)
            vectors.extend(np.asarray(matrix, dtype=np.float32))
        return vectors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["Indexer", "IngestReport"]
