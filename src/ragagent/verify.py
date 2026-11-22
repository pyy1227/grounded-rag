"""答不出就拒：用模型校验「仅凭检索到的段落能否回答」。

为什么需要它：
词面信号（未见实词比例、检索分数、覆盖率）在这套口语化评测集上都试过，
两类样本分布重叠，最好的工作点也要误拒 >50%——因为它们度量的是**措辞**，
而不是**资料里到底有没有答案**。语义判断是这类问题的正解。

离线（没有 API Key）时本模块不启用，Retriever 会退回词面信号。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Protocol, Tuple

from .jsonutil import load_lenient, truncate
from .llm import ChatModel
from .models import SearchHit

SYSTEM = "你是客服知识库的质检员，只输出 JSON。"

PROMPT = (
    "判断仅凭下面这些资料，能否回答用户的问题。\n"
    "规则：资料里必须有明确依据；只有相关话题但没有具体答案，算不能回答；\n"
    "能回答输出 {{\"answerable\": true, \"reason\": \"一句话依据\"}}，"
    "否则输出 {{\"answerable\": false, \"reason\": \"缺什么\"}}。\n\n"
    "问题：{query}\n\n资料：\n{context}"
)


class AnswerabilityVerifier(Protocol):
    name: str

    def verify(self, query: str, hits: List[SearchHit]) -> Tuple[bool, str]: ...


class LLMVerifier:
    """一次调用判定能否回答；结果按 (问题, 前 k 个 chunk) 缓存，避免评测重复计费。"""

    name = "llm"

    def __init__(self, chat: ChatModel, snippets: int = 3, max_chars: int = 300) -> None:
        self.chat = chat
        self.snippets = snippets
        self.max_chars = max_chars
        self.cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[bool, str]] = {}
        self.calls = 0

    def verify(self, query: str, hits: List[SearchHit]) -> Tuple[bool, str]:
        if not hits:
            return False, "没有检索到任何内容"

        key = (query, tuple(hit.chunk.chunk_id for hit in hits[: self.snippets]))
        if key in self.cache:
            return self.cache[key]

        context = "\n".join(
            f"[{index}] {truncate(hit.chunk.text, self.max_chars)}"
            for index, hit in enumerate(hits[: self.snippets], start=1)
        )
        self.calls += 1
        try:
            result = self.chat.chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT.format(query=query, context=context)},
                ],
                tools=None,
                temperature=0.0,
                max_tokens=200,
            )
            payload = load_lenient(result.content) or {}
            if "answerable" not in payload:
                raise ValueError("返回里没有 answerable 字段")
            verdict = (bool(payload["answerable"]), str(payload.get("reason") or ""))
        except Exception as exc:  # 校验失败时不当作拒答，避免把故障变成"不知道"
            verdict = (True, f"校验失败，按可答处理：{type(exc).__name__}")

        self.cache[key] = verdict
        return verdict


def build_verifier(chat: Optional[ChatModel], snippets: int = 3) -> Optional[LLMVerifier]:
    return LLMVerifier(chat, snippets=snippets) if chat is not None else None


__all__ = ["AnswerabilityVerifier", "LLMVerifier", "build_verifier"]
