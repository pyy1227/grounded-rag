"""向量检索：内存矩阵 + 余弦相似度。

数据量到十万级时应换 pgvector / Qdrant，这里的接口保持
`search(query_vector, top_k) -> [(chunk_id, score)]`，替换实现不影响上层。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class VectorIndex:
    def __init__(self, chunk_ids: List[str], matrix: np.ndarray) -> None:
        if len(chunk_ids) != len(matrix):
            raise ValueError("chunk 数量与向量数量不一致")
        self.chunk_ids = chunk_ids
        self.matrix = _normalize(matrix.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"查询向量维度不匹配: {query.shape[0]} != {self.matrix.shape[1]}")
        query = query / (np.linalg.norm(query) or 1.0)
        scores = self.matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    @classmethod
    def from_embeddings(cls, vectors: Dict[str, np.ndarray]) -> "VectorIndex":
        chunk_ids = list(vectors.keys())
        if not chunk_ids:
            return cls([], np.zeros((0, 0), dtype=np.float32))
        matrix = np.vstack([vectors[cid] for cid in chunk_ids])
        return cls(chunk_ids, matrix)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


__all__ = ["VectorIndex"]
