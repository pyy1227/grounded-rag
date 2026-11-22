"""Agent 主循环。

模型决策、调用工具、结果回灌，直到 submit_answer 或步数用尽。
所有动作记入 trace，便于回放和定位答错的那一步。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..generation import REFUSE_TEXT, AnswerGenerator
from ..llm import ChatModel, build_chat
from ..models import Answer, Citation, SearchHit
from ..retrieval.retriever import Retriever
from ..storage import Storage
from ..verify import build_grounding
from .prompts import AGENT_SYSTEM_PROMPT
from .tools import AgentContext, ToolRegistry


@dataclass
class AgentRun:
    answer: Answer
    steps: int
    trace: List[Dict[str, Any]]


class AgentRunner:
    """最小的 function-calling Agent。"""

    def __init__(
        self,
        storage: Storage,
        config: AppConfig,
        chat: Optional[ChatModel] = None,
        retriever: Optional[Retriever] = None,
        grounding: Optional[Any] = None,
    ) -> None:
        self.storage = storage
        self.config = config
        self.chat = chat or build_chat(config)
        self.retriever = retriever or Retriever(storage, config, chat=self.chat)
        self.registry = ToolRegistry()
        self.max_steps = max(1, int(config.agent.max_steps))
        self.require_citation = bool(config.agent.require_citation)
        # 和 /chat 共用同一层接地校验：Agent 的答案也要有引用段落支持
        self.grounding = grounding
        if self.grounding is None and config.retrieval.grounding_mode == "llm" and config.llm_api_key():
            self.grounding = build_grounding(self.chat)

    def run(self, query: str) -> Answer:
        started = time.perf_counter()
        query = str(query or "").strip()
        if not query:
            return Answer(text=REFUSE_TEXT, refused=True, mode="agent", trace=[])

        ctx = AgentContext(
            query=query,
            storage=self.storage,
            config=self.config,
            retriever=self.retriever,
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(max_steps=self.max_steps)},
            {"role": "user", "content": query},
        ]

        final_text = ""
        steps = 0

        for step in range(1, self.max_steps + 1):
            steps = step
            step_started = time.perf_counter()
            try:
                result = self.chat.chat(
                    messages,
                    tools=self.registry.schemas(),
                    temperature=0.0,
                )
            except Exception as exc:
                ctx.trace.append({"step": step, "action": "llm_error", "error": str(exc)})
                break

            if result.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.content or "",
                        "tool_calls": [call.to_message() for call in result.tool_calls],
                    }
                )
                for call in result.tool_calls:
                    tool_started = time.perf_counter()
                    output = self.registry.call(ctx, call.name, call.arguments)
                    ctx.trace.append(
                        {
                            "step": step,
                            "action": "tool",
                            "tool": call.name,
                            "arguments": call.arguments[:300],
                            "result": output[:400],
                            "latency_ms": int((time.perf_counter() - tool_started) * 1000),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": output,
                        }
                    )
                    if ctx.submitted is not None:
                        break
                if ctx.submitted is not None:
                    break
                continue

            final_text = result.content or ""
            ctx.trace.append({"step": step, "action": "answer", "content": final_text[:300]})
            break

        answer = self._finalize(ctx, final_text, started)
        answer.trace = ctx.trace
        return answer

    def _finalize(self, ctx: AgentContext, final_text: str, started: float) -> Answer:
        latency = int((time.perf_counter() - started) * 1000)

        if ctx.submitted:
            submitted = ctx.submitted
            text = str(submitted.get("answer") or "").strip()
            refused = bool(submitted.get("refused"))
            citations = self._citations_from_ids(ctx, submitted.get("citations") or [])
            if not citations and ctx.hits and self.require_citation:
                citations = self._auto_citations(ctx)
                ctx.trace.append({"action": "auto_cite", "note": "模型未给出有效引用，使用检索结果回填"})
            if refused or not text:
                return Answer(
                    text=text or REFUSE_TEXT,
                    citations=[],
                    refused=True,
                    mode="agent",
                    latency_ms=latency,
                )
            if not citations and (self.require_citation or not ctx.hits):
                return self._refuse(latency, ctx, "missing_citation")
            grounded, note = self._check_grounding(ctx, text, citations)
            if grounded is False:
                ctx.trace.append({"action": "grounding_failed", "note": note})
                return self._fallback(ctx, citations, latency, "答案未被引用资料支持，已降级为摘录")
            return Answer(
                text=text,
                citations=citations,
                refused=False,
                mode="agent",
                latency_ms=latency,
                grounded=grounded,
            )

        if final_text.strip():
            citations = self._auto_citations(ctx) if (ctx.hits and self.require_citation) else []
            if not citations and (self.require_citation or not ctx.hits):
                # 没检索就作答：按拒答处理，避免无依据回答
                return self._refuse(latency, ctx, "missing_citation")
            grounded, note = self._check_grounding(ctx, final_text.strip(), citations)
            if grounded is False:
                ctx.trace.append({"action": "grounding_failed", "note": note})
                return self._fallback(ctx, citations, latency, "答案未被引用资料支持，已降级为摘录")
            return Answer(
                text=final_text.strip(),
                citations=citations,
                refused=False,
                mode="agent",
                latency_ms=latency,
                grounded=grounded,
            )

        # 步数用尽且没有提交：用已有检索结果做摘录式兜底
        return self._fallback(ctx, self._auto_citations(ctx), latency)

    def _refuse(self, latency: int, ctx: AgentContext, reason: str) -> Answer:
        ctx.trace.append({"action": "refused", "note": reason})
        return Answer(text=REFUSE_TEXT, refused=True, mode="agent", latency_ms=latency)

    def _fallback(
        self,
        ctx: AgentContext,
        citations: List[Citation],
        latency: int,
        note: str = "",
    ) -> Answer:
        generator = AnswerGenerator(self.storage, self.config, self.retriever, self.chat)
        fallback = generator._extractive_answer(
            ctx.hits, citations or self._auto_citations(ctx), note=note
        )
        fallback.mode = "agent-fallback"
        fallback.latency_ms = latency
        if note and not fallback.refused:
            fallback.grounded = False
        return fallback

    def _check_grounding(
        self,
        ctx: AgentContext,
        text: str,
        citations: List[Citation],
    ) -> tuple[Optional[bool], str]:
        """Agent 答案也要过接地校验；没启用校验时返回 None（未知）。"""
        if self.grounding is None:
            return None, ""

        contexts: List[str] = []
        for citation in citations[:3]:
            chunk = self.storage.get_chunk(citation.chunk_id)
            if chunk is None:
                continue
            contexts.append(
                self.retriever.expand(SearchHit(chunk=chunk, score=0.0), max_chars=800)
            )
        if not contexts:
            contexts = [self.retriever.expand(hit, max_chars=800) for hit in ctx.hits[:3]]
        if not contexts:
            return False, "没有可用的引用资料"
        return self.grounding.check(text, contexts)

    def _citations_from_ids(self, ctx: AgentContext, chunk_ids: List[str]) -> List[Citation]:
        chunks = self.storage.get_chunks(chunk_ids)
        citations: List[Citation] = []
        for index, chunk_id in enumerate(chunk_ids, start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    index=index,
                    chunk_id=chunk.chunk_id,
                    doc_title=chunk.doc_id,
                    section_path=chunk.section_path,
                    snippet=chunk.text[:160].replace("\n", " "),
                )
            )
        return citations

    def _auto_citations(self, ctx: AgentContext) -> List[Citation]:
        citations: List[Citation] = []
        for index, hit in enumerate(ctx.hits[:3], start=1):
            citations.append(
                Citation(
                    index=index,
                    chunk_id=hit.chunk.chunk_id,
                    doc_title=hit.chunk.doc_id,
                    section_path=hit.chunk.section_path,
                    snippet=self.retriever.expand(hit, max_chars=160).replace("\n", " "),
                )
            )
        return citations


__all__ = ["AgentRunner", "AgentRun"]
