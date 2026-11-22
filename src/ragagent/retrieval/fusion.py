"""多路召回融合：RRF 与加权融合。

- RRF（Reciprocal Rank Fusion）：只看排名，不受各源分数量纲影响，鲁棒；
- 加权融合：把每路分数 min-max 归一后加权，需要调权重，但可解释性强。

两种都实现，评测时用同一套数据对比（见 evaluation.py）。
"""
from __future__ import annotations

from typing import Dict, List


def rrf_fuse(
    rank_lists: Dict[str, List[str]],
    k: int = 60,
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """rank_lists: {source: [chunk_id, ...]}（按分数降序）。返回归一化到 [0,1] 的融合分。

    weights 用于给不同召回源加权：BM25 的排名通常比哈希向量可靠，
    不加权时三路等权会把单路强命中的结果挤下去。
    """
    raw: Dict[str, float] = {}
    for source, ids in rank_lists.items():
        weight = 1.0 if weights is None else float(weights.get(source, 1.0))
        if weight <= 0:
            continue
        for rank, chunk_id in enumerate(ids, start=1):
            raw[chunk_id] = raw.get(chunk_id, 0.0) + weight / (k + rank)
    return _normalize(raw)


def weighted_fuse(
    score_lists: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """把每路分数 min-max 归一后按权重求和。"""
    raw: Dict[str, float] = {}
    for source, scores in score_lists.items():
        weight = float(weights.get(source, 0.0))
        if weight <= 0 or not scores:
            continue
        normalized = _minmax(scores)
        for chunk_id, score in normalized.items():
            raw[chunk_id] = raw.get(chunk_id, 0.0) + weight * score
    return _normalize(raw)


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high - low < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    high = max(scores.values())
    if high <= 0:
        return scores
    return {key: value / high for key, value in scores.items()}


__all__ = ["rrf_fuse", "weighted_fuse"]
