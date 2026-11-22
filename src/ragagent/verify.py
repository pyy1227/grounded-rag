"""拒答与接地校验。

AnswerabilityVerifier 在回答前判断检索到的资料够不够，分三档：
answerable 直接作答，need_more 由 Retriever 扩大窗口再检索一次，cannot 拒答。
need_more 单独一档，是为了区分资料确实没有和这次没检索到。

GroundingVerifier 在生成后检查答案里的每条主张能否在引用段落中找到依据，
找不到就降级为摘录。

没有 API Key 时两者都不启用，Retriever 退回词面信号。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Protocol, Tuple

from .jsonutil import load_lenient, truncate
from .llm import ChatModel
from .models import SearchHit

# 三档判定结果
ANSWERABLE = "answerable"
NEED_MORE = "need_more"
CANNOT = "cannot"

SYSTEM = "你是客服知识库的质检员，只输出 JSON。"

TIERED_PROMPT = (
    "判断仅凭下面这些资料，能否回答用户的问题。按三档回答：\n"
    "- answerable：资料能回答问题的**核心诉求**。答得不全、缺少细节也算这一档；\n"
    "- need_more：资料与问题相关，但缺少回答所**必需的关键事实**，可能还有没给出的段落；\n"
    "- cannot：这些资料与问题无关，或确实没有该信息。\n"
    "注意：只要资料里能直接找到答案的要素，就选 answerable；只有连核心事实都找不到时才选 need_more。\n"
    '只输出 JSON：{{"verdict": "answerable|need_more|cannot", "reason": "一句话"}}\n\n'
    "问题：{query}\n\n资料：\n{context}"
)

GROUNDING_PROMPT = (
    "下面是一段客服回答及其引用资料。检查回答里的每一条事实主张是否都能在资料中找到依据。\n"
    "要求：\n"
    "- 数字、期限、金额、条件必须与资料一致；\n"
    "- 资料没有提到的内容，视为没有依据；\n"
    "- 合理的同义改写不算编造。\n"
    '只输出 JSON：{{"supported": true/false, "unsupported": ["原文片段"], "reason": "一句话"}}\n\n'
    "回答：{answer}\n\n资料：\n{context}"
)


class AnswerabilityVerifier(Protocol):
    name: str

    def verify(self, query: str, hits: List[SearchHit]) -> Tuple[bool, str]: ...


class GroundingCheck(Protocol):
    name: str

    def check(self, answer: str, contexts: List[str]) -> Tuple[bool, str]: ...


class LLMVerifier:
    """三档判定；结果按 (问题, 前 k 个 chunk) 缓存，避免评测重复计费。"""

    name = "llm"

    def __init__(self, chat: ChatModel, snippets: int = 5, max_chars: int = 300) -> None:
        self.chat = chat
        self.snippets = snippets
        self.max_chars = max_chars
        self.cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]] = {}
        self.calls = 0

    def verify_tiered(self, query: str, hits: List[SearchHit]) -> Tuple[str, str]:
        if not hits:
            return CANNOT, "没有检索到任何内容"

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
                    {"role": "user", "content": TIERED_PROMPT.format(query=query, context=context)},
                ],
                tools=None,
                temperature=0.0,
                max_tokens=200,
            )
            payload = load_lenient(result.content) or {}
            verdict = str(payload.get("verdict") or "").strip().lower()
            if verdict not in (ANSWERABLE, NEED_MORE, CANNOT):
                # 老格式兼容：{"answerable": true}
                if "answerable" in payload:
                    verdict = ANSWERABLE if payload["answerable"] else CANNOT
                else:
                    raise ValueError(f"无法识别的判定：{verdict!r}")
            outcome = (verdict, str(payload.get("reason") or ""))
        except Exception as exc:  # 校验失败按可答处理，不让故障变成拒答
            outcome = (ANSWERABLE, f"校验失败，按可答处理：{type(exc).__name__}")

        self.cache[key] = outcome
        return outcome

    def verify(self, query: str, hits: List[SearchHit]) -> Tuple[bool, str]:
        verdict, reason = self.verify_tiered(query, hits)
        return verdict == ANSWERABLE, reason


class GroundingVerifier:
    """答案接地校验：每条主张是否被引用段落支持。"""

    name = "llm"

    def __init__(self, chat: ChatModel, max_chars: int = 800) -> None:
        self.chat = chat
        self.max_chars = max_chars
        self.calls = 0

    def check(self, answer: str, contexts: List[str]) -> Tuple[bool, str]:
        if not contexts or not answer.strip():
            return False, "没有可用的引用资料"
        context = "\n\n".join(
            f"[{index}] {truncate(text, self.max_chars)}"
            for index, text in enumerate(contexts, start=1)
        )
        self.calls += 1
        try:
            result = self.chat.chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": GROUNDING_PROMPT.format(answer=answer, context=context)},
                ],
                tools=None,
                temperature=0.0,
                max_tokens=300,
            )
            payload = load_lenient(result.content) or {}
            if "supported" not in payload:
                raise ValueError("返回里没有 supported 字段")
            unsupported = payload.get("unsupported") or []
            reason = str(payload.get("reason") or "")
            if unsupported:
                reason = f"{reason}；无依据片段：{unsupported[:2]}"
            return bool(payload["supported"]), reason
        except Exception as exc:  # 校验失败不拦截回答
            return True, f"接地校验失败，按通过处理：{type(exc).__name__}"


def build_verifier(chat: Optional[ChatModel], snippets: int = 5) -> Optional[LLMVerifier]:
    return LLMVerifier(chat, snippets=snippets) if chat is not None else None


def build_grounding(chat: Optional[ChatModel]) -> Optional[GroundingVerifier]:
    return GroundingVerifier(chat) if chat is not None else None


__all__ = [
    "ANSWERABLE", "NEED_MORE", "CANNOT",
    "AnswerabilityVerifier", "GroundingCheck", "LLMVerifier", "GroundingVerifier",
    "build_verifier", "build_grounding",
]
