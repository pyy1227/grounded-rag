"""切分策略测试：FAQ / 表格 / 父块。"""
from __future__ import annotations

from ragagent.chunking import chunk_document
from ragagent.config import ChunkingConfig
from ragagent.models import ParsedDocument, ParsedSection


def _doc(sections):
    return ParsedDocument(doc_id="d", title="t", path="d.md", sections=sections)


def test_faq_pairs_are_not_split():
    doc = _doc([
        ParsedSection(
            title="常见问题",
            level=1,
            text="Q：退货运费谁承担？\nA：质量问题商家承担。\n\nQ：退款多久到账？\nA：1-3 个工作日。",
        )
    ])
    chunks = chunk_document(doc, ChunkingConfig(max_chars=500, overlap_chars=20))
    faq = [c for c in chunks if c.kind == "faq"]
    assert len(faq) == 2
    assert "退货运费" in faq[0].text
    assert faq[0].text.count("Q：") == 1


def test_table_kept_as_one_chunk():
    table = "| 原因 | 承担 |\n| --- | --- |\n| 质量问题 | 商家 |\n| 无理由 | 买家 |"
    doc = _doc([ParsedSection(title="运费", level=1, text=table, kind="table")])
    chunks = chunk_document(doc, ChunkingConfig(max_chars=500, overlap_chars=20))
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert "质量问题" in chunks[0].text


def test_long_section_creates_parent_and_children():
    long_text = "\n\n".join(f"第{i}段内容，说明售后流程中的注意事项。" * 3 for i in range(12))
    doc = _doc([ParsedSection(title="长章节", level=1, text=long_text)])
    chunks = chunk_document(doc, ChunkingConfig(max_chars=200, overlap_chars=20, parent_max_chars=1200))
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert len(parents) == 1
    assert len(children) > 1
    assert all(child.parent_id == parents[0].chunk_id for child in children)


def test_section_path_keeps_hierarchy():
    doc = _doc([
        ParsedSection(title="售后政策", level=1, text="总述"),
        ParsedSection(title="退货运费", level=2, text="质量问题由商家承担。"),
    ])
    chunks = chunk_document(doc, ChunkingConfig(max_chars=500, overlap_chars=20))
    assert chunks[-1].section_path == "售后政策 > 退货运费"
