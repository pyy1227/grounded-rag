"""结构化切分。

策略（按优先级）：
1. FAQ：识别 Q/A 成对内容，一问一答一个 chunk，避免被切断；
2. 表格：整表一个 chunk（过长按行拆，保留表头）；
3. 正文：按段落贪心打包到 max_chars，带 overlap；
4. 父子块：章节过长时生成 parent chunk，子块检索命中后回填父块上下文。
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from .config import ChunkingConfig
from .models import Chunk, ParsedDocument, ParsedSection

FAQ_Q_RE = re.compile(r"^\s*(?:Q|问)\s*[:：]\s*(.+)$", re.IGNORECASE)
FAQ_A_RE = re.compile(r"^\s*(?:A|答)\s*[:：]\s*(.+)$", re.IGNORECASE)


def chunk_document(doc: ParsedDocument, cfg: ChunkingConfig) -> List[Chunk]:
    """把一篇文档切成 chunk 列表（含 parent）。

    父块按「章节」生成：章节总长度超过 max_chars 且被拆成多块时，
    生成一个 parent chunk，所有子块指向它，检索命中子块后回填父块上下文。
    """
    chunks: List[Chunk] = []
    ordinal = 0
    stack: List[tuple[int, str]] = []

    for section in doc.sections:
        while stack and stack[-1][0] >= section.level:
            stack.pop()
        stack.append((section.level, section.title))
        section_path = " > ".join(title for _, title in stack)

        pieces = _ensure_size(_split_section(section, cfg), cfg)
        if not pieces:
            continue

        parent_id: Optional[str] = None
        if len(section.text) > cfg.max_chars and len(pieces) > 1:
            ordinal += 1
            parent_id = _new_id(doc.doc_id, section_path, ordinal, "parent:" + section.text[:80])
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    section_path=section_path,
                    kind=section.kind,
                    text=section.text[: cfg.parent_max_chars],
                    is_parent=True,
                    parent_id=None,
                    meta=dict(section.meta),
                    )
            )

        header = f"（章节：{section_path}）\n"
        for text, kind in pieces:
            ordinal += 1
            chunk_text = header + text
            chunks.append(
                Chunk(
                    chunk_id=_new_id(doc.doc_id, section_path, ordinal, chunk_text),
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    section_path=section_path,
                    kind=kind,
                    text=chunk_text,
                    is_parent=False,
                    parent_id=parent_id,
                    meta=dict(section.meta),
                )
            )
    return chunks


def _ensure_size(pieces: List[tuple[str, str]], cfg: ChunkingConfig) -> List[tuple[str, str]]:
    """FAQ 或表格的单块可能超长，这里再做一次兜底切分。"""
    result: List[tuple[str, str]] = []
    for text, kind in pieces:
        if len(text) <= cfg.max_chars:
            result.append((text, kind))
            continue
        for part in _split_long(text, cfg.max_chars, cfg.overlap_chars):
            result.append((part, kind))
    return result



def _split_section(section: ParsedSection, cfg: ChunkingConfig) -> List[tuple[str, str]]:
    """返回 [(text, kind), ...]。"""
    text = section.text.strip()
    if not text:
        return []

    if section.kind == "table":
        return [(piece, "table") for piece in _split_table(text, cfg.max_chars)]

    faq_pairs = _extract_faq(text)
    if len(faq_pairs) >= 2:
        return [(pair, "faq") for pair in faq_pairs]

    return [(piece, "prose") for piece in _pack_paragraphs(text, cfg.max_chars, cfg.overlap_chars)]


def _extract_faq(text: str) -> List[str]:
    """把连续 Q/A 识别成问答对；识别不到返回空列表。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pairs: List[str] = []
    index = 0
    while index < len(lines):
        q_match = FAQ_Q_RE.match(lines[index])
        if q_match and index + 1 < len(lines):
            a_match = FAQ_A_RE.match(lines[index + 1])
            if a_match:
                pairs.append(f"Q：{q_match.group(1).strip()}\nA：{a_match.group(1).strip()}")
                index += 2
                continue
        index += 1
    return pairs


def _split_table(text: str, max_chars: int) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(text) <= max_chars or len(lines) <= 3:
        return [text]

    header = lines[:2]
    body = lines[2:]
    pieces: List[str] = []
    buffer: List[str] = []
    size = sum(len(line) for line in header)
    for row in body:
        if buffer and size + len(row) > max_chars:
            pieces.append("\n".join(header + buffer))
            buffer = []
            size = sum(len(line) for line in header)
        buffer.append(row)
        size += len(row)
    if buffer:
        pieces.append("\n".join(header + buffer))
    return pieces or [text]


def _pack_paragraphs(text: str, max_chars: int, overlap: int) -> List[str]:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []

    pieces: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_split_long(paragraph, max_chars, overlap))
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            tail = buffer[-overlap:] if buffer and overlap > 0 else ""
            buffer = (tail + "\n\n" + paragraph).strip() if tail else paragraph
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_long(text: str, max_chars: int, overlap: int) -> List[str]:
    pieces: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        pieces.append(text[start : start + max_chars])
        start += step
    return pieces


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", "\n", "；"):
        position = cut.rfind(sep)
        if position > limit * 0.6:
            return cut[: position + 1]
    return cut


def _new_id(doc_id: str, section_path: str, ordinal: int, text: str) -> str:
    raw = f"{doc_id}|{section_path}|{ordinal}|{text[:80]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_stats(chunks: List[Chunk]) -> Dict[str, int]:
    return {
        "total": len(chunks),
        "children": sum(1 for c in chunks if not c.is_parent),
        "parents": sum(1 for c in chunks if c.is_parent),
        "faq": sum(1 for c in chunks if c.kind == "faq"),
        "table": sum(1 for c in chunks if c.kind == "table"),
        "avg_chars": int(sum(len(c.text) for c in chunks) / max(1, len(chunks))),
    }


__all__ = ["chunk_document", "chunk_stats"]
