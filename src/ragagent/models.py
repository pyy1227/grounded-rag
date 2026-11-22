"""核心数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedSection:
    """文档解析后的一个章节。"""

    title: str
    level: int
    text: str
    kind: str = "prose"  # prose | table | faq
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """一篇文档的解析结果。"""

    doc_id: str
    title: str
    path: str
    sections: List[ParsedSection]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """入库的最小检索单元。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    section_path: str
    kind: str
    text: str
    is_parent: bool = False
    parent_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple:
        import json

        return (
            self.chunk_id,
            self.doc_id,
            self.ordinal,
            self.section_path,
            self.kind,
            self.text,
            1 if self.is_parent else 0,
            self.parent_id,
            json.dumps(self.meta, ensure_ascii=False),
        )


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    path: str
    sha256: str
    size: int
    mtime: float
    ingested_at: str
    version: int
    chunk_count: int = 0


@dataclass
class SearchHit:
    """一次检索命中的结果，保留各阶段分数便于解释。"""

    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section_path,
            "kind": self.chunk.kind,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "bm25": round(self.bm25_score, 4),
            "vector": round(self.vector_score, 4),
            "fused": round(self.fused_score, 4),
            "rerank": round(self.rerank_score, 4),
            "reasons": self.reasons,
        }


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_title: str
    section_path: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "section": self.section_path,
            "snippet": self.snippet,
        }


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False
    mode: str = "llm"  # llm | extractive
    latency_ms: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "refused": self.refused,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
            "trace": self.trace,
        }


@dataclass
class EvalCase:
    query: str
    gold: List[str] = field(default_factory=list)  # 形如 "doc_id::section 关键字"
    expected_keywords: List[str] = field(default_factory=list)
    answerable: bool = True


@dataclass
class CaseResult:
    case: EvalCase
    hit_keys: List[str] = field(default_factory=list)
    ranks: List[int] = field(default_factory=list)
    refused: bool = False
    answer_hit: bool = False


__all__ = [
    "ParsedSection", "ParsedDocument", "Chunk", "DocumentRecord", "SearchHit",
    "Citation", "Answer", "EvalCase", "CaseResult",
]
