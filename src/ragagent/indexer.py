"""入库流水线：解析、切分、向量化、落库。

文件 sha256 没变且没给 --force 就跳过，避免重复 embedding 计费；更新时整篇替换 chunk，
不留旧块；先算完 chunk 和向量再写库，embedding 失败不会破坏已有索引。
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
from .models import Chunk, DocumentRecord
from .parsing import SUPPORTED_SUFFIXES, parse_document
from .storage import Storage
from .tokenization import mode as tokenizer_mode

# 单条 embedding 输入的字符上限（text-embedding-v3 单条 token 上限约 8192，
# 中文按 1 字≈1 token 估，留足余量）。正常 chunk 只有几百字，不会触发。
_EMBED_MAX_CHARS = 2000


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

        if not files:
            raise ValueError(f"没有可入库的文件: {target}（支持 {sorted(SUPPORTED_SUFFIXES)}）")

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
            # documents 只数成功处理的文档（added/updated/skipped），失败的不算
            "documents": len(report.added) + len(report.updated) + len(report.skipped),
            "attempted": len(files),
            "errors": len(report.errors),
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

    def ingest_records(self, records: Iterable[dict]) -> dict:
        """一条记录当一个 chunk 批量入库，给外部公开评测集用。

        外部数据集本身就是 query 和 passage 配对，passage 不用再解析切分，
        所以直接建 chunk、算向量、一次事务写库。
        记录格式：{"doc_id": "...", "title": "...", "text": "..."}
        """
        started = time.perf_counter()
        report = IngestReport()
        prepared: List[tuple[str, str, str]] = []
        for record in records:
            doc_id = str(record.get("doc_id") or "").strip()
            text = str(record.get("text") or "").strip()
            if not doc_id or not text:
                report.errors.append({"path": doc_id or "<empty>", "error": "缺少 doc_id 或 text"})
                continue
            prepared.append((doc_id, str(record.get("title") or "passage"), text))

        if not prepared:
            raise ValueError("没有可入库的记录")

        now = datetime.now(timezone.utc).isoformat()
        # 外部语料里可能夹着超长 passage（实测最长 6.8 万字），直接送 embedding 会被
        # 服务端 400 整批打回，所以截断后再算向量，正文仍完整入库。
        matrix = self._embed([text[:_EMBED_MAX_CHARS] for _, _, text in prepared])
        chunks: List[Chunk] = []
        documents: List[DocumentRecord] = []
        vectors = {}
        for (doc_id, title, text), vector in zip(prepared, matrix):
            chunk = Chunk(
                chunk_id=doc_id, doc_id=doc_id, ordinal=0,
                section_path=title, kind="passage", text=text,
            )
            chunks.append(chunk)
            vectors[chunk.chunk_id] = vector
            documents.append(
                DocumentRecord(
                    doc_id=doc_id, title=title, path=f"external:{doc_id}",
                    sha256=_sha256_text(text), size=len(text.encode("utf-8")),
                    mtime=0.0, ingested_at=now, version=1, chunk_count=1,
                )
            )

        self.storage.replace_bulk(chunks, self.embedder.model, vectors, documents)
        self.storage.set_meta("last_ingest_at", now)

        report.added = [doc.doc_id for doc in documents]
        report.chunks = len(chunks)
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        report.stats = {
            "documents": len(documents),
            "children": len(chunks),
            "embedder": self.embedder.model,
            "tokenizer": tokenizer_mode(),
        }
        return report.to_dict()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["Indexer", "IngestReport"]
