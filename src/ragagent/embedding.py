"""Embedding 提供方。

RemoteEmbedder 调 OpenAI 兼容的 /embeddings，语义质量好，生产用。
HashEmbedder 是本地确定性哈希向量，只能离线跑通和单测，没有语义能力。

模型名里带维度和分词模式（如 hash-256-bigram），换模型或换分词器后新旧向量不会混用。
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .config import AppConfig
from .tokenization import mode as tokenizer_mode
from .tokenization import tokenize as _tokenize


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """把 token 哈希到固定维度，带符号累加后做 L2 归一化。

    只能捕捉字面重叠，不理解同义；作用是让整条链路在没有外部服务时也能跑通。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        # 模型名带上分词模式，换分词器后向量不兼容。必须和 tokenization 用同一套分词，
        # 否则名字写 bigram、实际按 jieba 切，换台机器就静默换了向量空间。
        self.model = f"hash-{dim}-{tokenizer_mode()}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        return _normalize(matrix)


class RemoteEmbedder:
    """OpenAI 兼容 /embeddings 客户端。"""

    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 10.0) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.dim = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(
            self.url,
            headers=headers,
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if len(data) != len(texts):
            raise RuntimeError(f"embedding 返回数量不匹配: {len(data)} != {len(texts)}")
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0))]
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = int(matrix.shape[1])
        return _normalize(matrix)


def build_embedder(config: AppConfig) -> Embedder:
    """按配置选择实现；remote 缺少 key 时也允许匿名调用（本地服务常见）。"""
    if config.embedding.provider == "remote":
        return RemoteEmbedder(
            url=config.embedding.url,
            model=config.embedding.model,
            api_key=config.embedding_api_key(),
            timeout=config.embedding.timeout,
        )
    return HashEmbedder(dim=config.embedding.dim)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


__all__ = ["Embedder", "HashEmbedder", "RemoteEmbedder", "build_embedder"]
