"""文档解析：把不同格式统一成「标题 + 章节」结构。

支持的格式：
- .md / .txt：按 Markdown 标题切章节；识别表格行
- .pdf：按页抽取文本（PyPDF2/pypdf）
- .docx：按标题样式切章节，表格转 Markdown
- .xlsx：每个 sheet 转一个 Markdown 表格

设计说明：解析层只负责「还原结构」，不做切分和清洗，
这样切分策略可以独立迭代（见 chunking.py）。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from .models import ParsedDocument, ParsedSection

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx", ".xlsx"}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def _doc_id(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    return rel.with_suffix("").as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_document(path: str | Path, root: str | Path | None = None) -> ParsedDocument:
    """解析单个文件。root 用来生成稳定的 doc_id。"""
    file_path = Path(path)
    root_path = Path(root) if root else file_path.parent
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")

    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        sections = _parse_text_markdown(file_path)
    elif suffix == ".pdf":
        sections = _parse_pdf(file_path)
    elif suffix == ".docx":
        sections = _parse_docx(file_path)
    else:
        sections = _parse_xlsx(file_path)

    sections = [s for s in sections if s.text.strip()]
    return ParsedDocument(
        doc_id=_doc_id(file_path, root_path),
        title=file_path.stem,
        path=str(file_path),
        sections=sections,
        meta={"sha256": _sha256(file_path), "suffix": suffix},
    )


def _parse_text_markdown(path: Path) -> List[ParsedSection]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    sections: List[ParsedSection] = []
    title = path.stem
    level = 1
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                ParsedSection(
                    title=title,
                    level=level,
                    text=body,
                    kind="table" if _looks_like_table(body) else "prose",
                )
            )
        buffer.clear()

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            continue
        buffer.append(line)

    flush()
    return sections


def _looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    table_lines = sum(1 for line in lines if TABLE_LINE_RE.match(line))
    return table_lines >= max(2, len(lines) * 0.6)


def _parse_pdf(path: Path) -> List[ParsedSection]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: List[ParsedSection] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _normalize_whitespace(text)
        if text.strip():
            sections.append(
                ParsedSection(
                    title=f"第 {index} 页",
                    level=1,
                    text=text,
                    meta={"page": index},
                )
            )
    return sections


def _parse_docx(path: Path) -> List[ParsedSection]:
    import docx

    document = docx.Document(str(path))
    sections: List[ParsedSection] = []
    title = path.stem
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                ParsedSection(
                    title=title,
                    level=1,
                    text=body,
                    kind="table" if _looks_like_table(body) else "prose",
                )
            )
        buffer.clear()

    for paragraph in document.paragraphs:
        style = (paragraph.style.name or "").lower() if paragraph.style else ""
        text = (paragraph.text or "").strip()
        if not text:
            continue
        if style.startswith("heading"):
            flush()
            title = text
            continue
        buffer.append(text)

    for table in document.tables:
        flush()
        sections.append(
            ParsedSection(
                title=title,
                level=1,
                text=_table_to_markdown(table),
                kind="table",
            )
        )

    flush()
    return sections


def _parse_xlsx(path: Path) -> List[ParsedSection]:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), data_only=True)
    sections: List[ParsedSection] = []
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if cell is None else str(cell).strip() for cell in row]
            if any(cells):
                rows.append(cells)
        if not rows:
            continue
        header = rows[0]
        body = rows[1:]
        markdown = _rows_to_markdown(header, body)
        sections.append(
            ParsedSection(
                title=f"{sheet.title}",
                level=1,
                text=markdown,
                kind="table",
                meta={"sheet": sheet.title, "rows": len(body)},
            )
        )
    return sections


def _table_to_markdown(table) -> str:
    rows = []
    for row in table.rows:
        rows.append([(cell.text or "").strip() for cell in row.cells])
    if not rows:
        return ""
    return _rows_to_markdown(rows[0], rows[1:])


def _rows_to_markdown(header: List[str], body: List[List[str]]) -> str:
    def fmt(cells: List[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    lines = [fmt(header), "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        row = (row + [""] * len(header))[: len(header)]
        lines.append(fmt(row))
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_directory(directory: str | Path) -> List[ParsedDocument]:
    """解析目录下所有支持的文件，按路径排序保证结果稳定。"""
    root = Path(directory)
    documents: List[ParsedDocument] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(parse_document(path, root))
    return documents


__all__ = ["parse_document", "parse_directory", "SUPPORTED_SUFFIXES"]
