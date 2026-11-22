"""RAG Agent 的工具集。

工具是 Agent 唯一的动作出口：检索、读全文、列文档、对比、提交答案。
参数用 Pydantic 定义，schema 直接喂给模型，校验在服务端做。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import AppConfig
from ..models import SearchHit
from ..retrieval.retriever import Retriever
from ..storage import Storage


class SearchArgs(BaseModel):
    query: str = Field(..., description="检索问题，使用客户的原始表述")
    top_k: int = Field(5, ge=1, le=10, description="返回条数")


class OpenChunkArgs(BaseModel):
    chunk_id: str = Field(..., description="search_knowledge 返回的 chunk_id")


class ListDocsArgs(BaseModel):
    limit: int = Field(20, ge=1, le=100)


class CompareArgs(BaseModel):
    doc_a: str = Field(..., description="文档 A 的 doc_id")
    doc_b: str = Field(..., description="文档 B 的 doc_id")
    aspect: str = Field(..., description="对比维度，例如「退货时效」")


class SubmitArgs(BaseModel):
    answer: str = Field(..., description="最终回答，必须基于工具结果")
    citations: List[str] = Field(default_factory=list, description="引用的 chunk_id 列表")
    refused: bool = Field(False, description="资料不足时置 true")


@dataclass
class ToolSpec:
    name: str
    description: str
    params: type[BaseModel]
    func: Callable[["AgentContext", BaseModel], str]

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params.model_json_schema(),
            },
        }


@dataclass
class AgentContext:
    query: str
    storage: Storage
    config: AppConfig
    retriever: Retriever
    hits: List[SearchHit] = field(default_factory=list)
    submitted: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


def _search(ctx: AgentContext, args: SearchArgs) -> str:
    hits = ctx.retriever.search(args.query, top_k=min(args.top_k, 8))
    ctx.hits = hits
    payload = [
        {
            "chunk_id": hit.chunk.chunk_id,
            "doc_id": hit.chunk.doc_id,
            "section": hit.chunk.section_path,
            "score": round(hit.score, 3),
            "snippet": ctx.retriever.expand(hit, max_chars=260).replace("\n", " "),
        }
        for hit in hits
    ]
    if not payload:
        return json.dumps({"results": [], "note": "没有检索到相关内容，可以换关键词或直接说明无法回答"}, ensure_ascii=False)
    return json.dumps({"results": payload}, ensure_ascii=False)


def _open_chunk(ctx: AgentContext, args: OpenChunkArgs) -> str:
    chunk = ctx.storage.get_chunk(args.chunk_id)
    if chunk is None:
        return json.dumps({"error": "chunk 不存在"}, ensure_ascii=False)
    text = chunk.text
    if chunk.parent_id:
        parent = ctx.storage.get_chunk(chunk.parent_id)
        if parent is not None:
            text = parent.text
    return json.dumps(
        {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "section": chunk.section_path, "text": text[:1600]},
        ensure_ascii=False,
    )


def _list_documents(ctx: AgentContext, args: ListDocsArgs) -> str:
    documents = ctx.storage.list_documents()[: args.limit]
    payload = [
        {"doc_id": doc.doc_id, "title": doc.title, "chunks": doc.chunk_count, "version": doc.version}
        for doc in documents
    ]
    return json.dumps({"documents": payload}, ensure_ascii=False)


def _compare(ctx: AgentContext, args: CompareArgs) -> str:
    def collect(doc_id: str) -> List[Dict[str, Any]]:
        hits = [hit for hit in ctx.retriever.search(args.aspect, top_k=8) if hit.chunk.doc_id == doc_id]
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "section": hit.chunk.section_path,
                "snippet": ctx.retriever.expand(hit, max_chars=300).replace("\n", " "),
            }
            for hit in hits[:2]
        ]

    return json.dumps(
        {"aspect": args.aspect, args.doc_a: collect(args.doc_a), args.doc_b: collect(args.doc_b)},
        ensure_ascii=False,
    )


def _submit(ctx: AgentContext, args: SubmitArgs) -> str:
    ctx.submitted = args.model_dump()
    return json.dumps({"ok": True}, ensure_ascii=False)


def build_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="search_knowledge",
            description="在知识库中检索与问题相关的片段。回答事实性问题前必须先调用。",
            params=SearchArgs,
            func=_search,
        ),
        ToolSpec(
            name="open_chunk",
            description="读取某个 chunk 的完整内容（会带出所属父块），用于确认细节。",
            params=OpenChunkArgs,
            func=_open_chunk,
        ),
        ToolSpec(
            name="list_documents",
            description="列出知识库里的文档，用于确认某个主题是否被收录。",
            params=ListDocsArgs,
            func=_list_documents,
        ),
        ToolSpec(
            name="compare_documents",
            description="按某个维度分别检索两篇文档并返回片段，用于做规则对比。",
            params=CompareArgs,
            func=_compare,
        ),
        ToolSpec(
            name="submit_answer",
            description="提交最终回答。answer 必须基于工具结果，citations 必须是真实 chunk_id；资料不足时 refused=true。",
            params=SubmitArgs,
            func=_submit,
        ),
    ]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools = {tool.name: tool for tool in build_tools()}

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools)

    def call(self, ctx: AgentContext, name: str, arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            args = tool.params.model_validate_json(arguments or "{}")
        except Exception as exc:
            return json.dumps({"error": f"参数不合法: {exc}"}, ensure_ascii=False)
        try:
            return tool.func(ctx, args)
        except Exception as exc:
            return json.dumps({"error": f"工具执行失败: {exc}"}, ensure_ascii=False)


__all__ = ["ToolRegistry", "ToolSpec", "AgentContext"]
