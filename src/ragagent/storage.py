"""SQLite 存储层。

不用 SQLAlchemy：只有 5 张表、查询固定，裸 sqlite3 更透明，也少一层依赖。
换 PostgreSQL + pgvector 时只需要改这个文件。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from .models import Chunk, DocumentRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    ingested_at TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    chunk_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    section_path TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    is_parent    INTEGER NOT NULL DEFAULT 0,
    parent_id    TEXT,
    meta_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_embeddings_model ON embeddings(model);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    report_path TEXT
);
"""


class Storage:
    """所有持久化操作的唯一入口。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # 文档
    def upsert_document(self, record: DocumentRecord) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, title, path, sha256, size, mtime, ingested_at, version, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title, path=excluded.path, sha256=excluded.sha256,
                    size=excluded.size, mtime=excluded.mtime, ingested_at=excluded.ingested_at,
                    version=excluded.version, chunk_count=excluded.chunk_count
                """,
                (
                    record.doc_id, record.title, record.path, record.sha256, record.size,
                    record.mtime, record.ingested_at, record.version, record.chunk_count,
                ),
            )

    def delete_document(self, doc_id: str) -> None:
        with self.conn() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._to_document(row) if row else None

    def find_document_by_path(self, path: str) -> Optional[DocumentRecord]:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        return self._to_document(row) if row else None

    def list_documents(self) -> List[DocumentRecord]:
        with self.conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [self._to_document(row) for row in rows]

    @staticmethod
    def _to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"], title=row["title"], path=row["path"], sha256=row["sha256"],
            size=row["size"], mtime=row["mtime"], ingested_at=row["ingested_at"],
            version=row["version"], chunk_count=row["chunk_count"],
        )

    # Chunk
    def replace_chunks(self, doc_id: str, chunks: Iterable[Chunk]) -> int:
        """整篇替换：先删旧 chunk/embedding，再写入。"""
        chunks = list(chunks)
        with self.conn() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, doc_id, ordinal, section_path, kind, text, is_parent, parent_id, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [chunk.to_row() for chunk in chunks],
            )
        return len(chunks)

    def replace_bulk(
        self,
        chunks: Iterable[Chunk],
        model: str,
        vectors: Dict[str, np.ndarray],
        documents: Iterable[DocumentRecord],
    ) -> int:
        """外部评测集入库用，一个连接内做完，避免几千次开关库。

        语义和 replace_chunks 一致：先按 doc_id 删旧，再写新；这里只是省掉逐篇开销。
        """
        chunks = list(chunks)
        documents = list(documents)
        doc_ids = [doc.doc_id for doc in documents]
        with self.conn() as conn:
            for doc_id in doc_ids:
                conn.execute(
                    "DELETE FROM embeddings WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                    (doc_id,),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, doc_id, ordinal, section_path, kind, text, is_parent, parent_id, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [chunk.to_row() for chunk in chunks],
            )
            payload = []
            for chunk in chunks:
                vector = vectors.get(chunk.chunk_id)
                if vector is None:
                    continue
                array = np.asarray(vector, dtype=np.float32)
                payload.append((chunk.chunk_id, model, int(array.shape[0]), array.tobytes()))
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                payload,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO documents(doc_id, title, path, sha256, size, mtime, ingested_at, version, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (doc.doc_id, doc.title, doc.path, doc.sha256, doc.size, doc.mtime,
                     doc.ingested_at, doc.version, doc.chunk_count)
                    for doc in documents
                ],
            )
        return len(chunks)

    def iter_chunks(self, include_parents: bool = False) -> List[Chunk]:
        sql = "SELECT * FROM chunks"
        if not include_parents:
            sql += " WHERE is_parent = 0"
        sql += " ORDER BY doc_id, ordinal"
        with self.conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Iterable[str]) -> Dict[str, Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.conn() as conn:
            rows = conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids).fetchall()
        return {row["chunk_id"]: self._to_chunk(row) for row in rows}

    def count_chunks(self, doc_id: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM chunks WHERE is_parent = 0"
        params: tuple = ()
        if doc_id:
            sql += " AND doc_id = ?"
            params = (doc_id,)
        with self.conn() as conn:
            return int(conn.execute(sql, params).fetchone()["n"])

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], ordinal=row["ordinal"],
            section_path=row["section_path"], kind=row["kind"], text=row["text"],
            is_parent=bool(row["is_parent"]), parent_id=row["parent_id"],
            meta=json.loads(row["meta_json"] or "{}"),
        )

    # 向量
    def save_embeddings(self, model: str, vectors: Dict[str, np.ndarray]) -> None:
        if not vectors:
            return
        payload = []
        for chunk_id, vector in vectors.items():
            array = np.asarray(vector, dtype=np.float32)
            payload.append((chunk_id, model, int(array.shape[0]), array.tobytes()))
        with self.conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                payload,
            )

    def load_embeddings(self, model: str) -> Dict[str, np.ndarray]:
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, dim, vector FROM embeddings WHERE model = ?", (model,)
            ).fetchall()
        result: Dict[str, np.ndarray] = {}
        for row in rows:
            result[row["chunk_id"]] = np.frombuffer(row["vector"], dtype=np.float32)
        return result

    def count_embeddings(self, model: Optional[str] = None) -> int:
        """向量条数。传 model 时只数该模型，用来发现配了向量但库里没有的情况。"""
        sql = "SELECT COUNT(*) AS n FROM embeddings"
        params: tuple = ()
        if model is not None:
            sql += " WHERE model = ?"
            params = (model,)
        with self.conn() as conn:
            return int(conn.execute(sql, params).fetchone()["n"])

    def list_embedding_models(self) -> Dict[str, int]:
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model"
            ).fetchall()
        return {row["model"]: int(row["n"]) for row in rows}

    # meta 和评测
    def set_meta(self, key: str, value: str) -> None:
        with self.conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def save_eval_run(
        self,
        run_id: str,
        dataset: str,
        config_json: Dict[str, Any],
        metrics: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs(run_id, created_at, dataset, config_json, metrics_json, report_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    dataset,
                    json.dumps(config_json, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    report_path,
                ),
            )

    def list_eval_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "dataset": row["dataset"],
                "metrics": json.loads(row["metrics_json"]),
                "report_path": row["report_path"],
            }
            for row in rows
        ]


__all__ = ["Storage", "SCHEMA"]
