"""精排。

三种实现：noop 按融合分排序；heuristic 用融合分、标题重合、短语命中、长度惩罚，
零成本、可解释；llm 把候选交给模型 listwise 打分，上限高，但多一次调用。
评测用同一份数据对比三者，决定线上用哪个。
"""
from __future__ import annotations

from typing import List, Optional, Protocol

from ..config import AppConfig
from ..jsonutil import load_lenient, truncate
from ..llm import ChatModel
from ..models import SearchHit
from ..tokenization import normalize, tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]: ...


class NoopReranker:
    name = "noop"

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        ordered = sorted(hits, key=lambda hit: hit.fused_score, reverse=True)[:top_k]
        for hit in ordered:
            hit.score = hit.fused_score
        return ordered


class HeuristicReranker:
    """不调模型，用可解释信号做二次排序。"""

    name = "heuristic"

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        query_tokens = set(tokenize(query))
        query_norm = normalize(query)
        scored: List[SearchHit] = []

        for hit in hits:
            text_norm = normalize(hit.chunk.text)
            section_tokens = set(tokenize(hit.chunk.section_path))
            overlap = (
                len(query_tokens & section_tokens) / len(query_tokens)
                if query_tokens else 0.0
            )
            phrase = 1.0 if len(query_norm) >= 2 and query_norm in text_norm else 0.0
            length = len(hit.chunk.text)
            length_score = max(0.0, 1.0 - abs(length - 320) / 1200)

            score = (
                0.55 * hit.fused_score
                + 0.20 * overlap
                + 0.15 * phrase
                + 0.10 * length_score
            )
            hit.rerank_score = round(score, 6)
            hit.score = hit.rerank_score
            if overlap > 0:
                hit.reasons.append(f"标题重合 {overlap:.2f}")
            if phrase:
                hit.reasons.append("原文短语命中")
            scored.append(hit)

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class LLMReranker:
    """listwise 打分：一次请求给所有候选打分，控制成本和延迟。"""

    name = "llm"

    def __init__(self, chat: ChatModel, fallback: Optional[Reranker] = None) -> None:
        self.chat = chat
        self.fallback = fallback or HeuristicReranker()

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            return []
        candidates = []
        for hit in hits[:20]:
            candidates.append(
                f"id={hit.chunk.chunk_id}｜章节={hit.chunk.section_path}｜{truncate(hit.chunk.text, 240)}"
            )
        prompt = (
            "你是检索精排器。请判断每个候选与问题的相关性，打分 0-3（3 最相关）。\n"
            "只输出 JSON：{\"scores\": [{\"id\": \"...\", \"score\": 2}]}\n\n"
            f"问题：{query}\n\n候选：\n" + "\n".join(candidates)
        )
        try:
            result = self.chat.chat(
                [{"role": "system", "content": "你只输出 JSON。"}, {"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
                max_tokens=400,
            )
            payload = load_lenient(result.content) or {}
            mapping = {
                str(item.get("id")): float(item.get("score", 0))
                for item in payload.get("scores", [])
                if isinstance(item, dict)
            }
            if not mapping:
                return self.fallback.rerank(query, hits, top_k)
            for hit in hits:
                score = mapping.get(hit.chunk.chunk_id, 0.0)
                hit.rerank_score = round(score / 3.0, 6)
                hit.score = hit.rerank_score
                hit.reasons.append("LLM 精排")
            ordered = sorted(hits, key=lambda item: item.score, reverse=True)
            return [hit for hit in ordered if hit.score > 0][:top_k] or self.fallback.rerank(query, hits, top_k)
        except Exception:
            return self.fallback.rerank(query, hits, top_k)


def build_reranker(config: AppConfig, chat: Optional[ChatModel] = None) -> Reranker:
    mode = config.retrieval.rerank_mode
    if mode == "noop" or not config.retrieval.enable_rerank:
        return NoopReranker()
    if mode == "llm" and chat is not None:
        return LLMReranker(chat)
    return HeuristicReranker()


__all__ = ["Reranker", "NoopReranker", "HeuristicReranker", "LLMReranker", "build_reranker"]
