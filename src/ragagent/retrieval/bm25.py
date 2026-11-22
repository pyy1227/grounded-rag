"""BM25 检索，自己实现，不引 rank_bm25。

自己写是为了分词可控、分数可解释，后面也方便做字段加权（section 标题和正文权重不同）。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..models import Chunk
from ..tokenization import tokenize


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """内存版 BM25。数据量在万级 chunk 以内足够，换 pgvector 时可替换。"""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = list(chunks)
        self._term_freqs: List[Counter] = []
        self._lengths: List[int] = []
        self._df: Counter = Counter()
        self._build()

    def _build(self) -> None:
        for chunk in self.chunks:
            # chunk.text 已带（章节：...）前缀，标题词自然参与打分，不再单独加权
            counts = Counter(tokenize(chunk.text))
            self._term_freqs.append(counts)
            self._lengths.append(max(1, sum(counts.values())))
            for token in counts:
                self._df[token] += 1

        self._avg_len = sum(self._lengths) / max(1, len(self._lengths))
        total = max(1, len(self.chunks))
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def df(self, token: str) -> int:
        """某个 token 出现在多少个 chunk 里（用于判断问题里的词是否见过）。"""
        return int(self._df.get(token, 0))

    def search(self, query: str, top_k: int = 20) -> List[BM25Hit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scores: List[Tuple[float, int]] = []
        for index, counts in enumerate(self._term_freqs):
            length = self._lengths[index]
            score = 0.0
            for token in query_tokens:
                freq = counts.get(token)
                if not freq:
                    continue
                idf = self._idf.get(token, 0.0)
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += idf * freq * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index))

        scores.sort(reverse=True)
        return [BM25Hit(chunk=self.chunks[index], score=score) for score, index in scores[:top_k]]


__all__ = ["BM25Index", "BM25Hit"]
