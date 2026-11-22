"""配置模型：TOML 文件 + 环境变量覆盖。

设计原则：
- 所有可调参数（切分、召回权重、阈值）都进配置，方便做消融实验；
- 密钥只从环境变量读，不落配置文件。
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class StorageConfig:
    path: str = "data/rag.db"


@dataclass
class ChunkingConfig:
    max_chars: int = 520
    overlap_chars: int = 80
    parent_max_chars: int = 1600
    min_chars: int = 40


@dataclass
class EmbeddingConfig:
    # hash = 离线确定性向量（仅用于跑通和测试）；remote = OpenAI 兼容 /embeddings
    provider: str = "hash"
    url: str = "http://127.0.0.1:8081/v1/embeddings"
    model: str = "bge-large-zh-v1.5"
    api_key_env: str = "RAGAGENT_EMBEDDING_API_KEY"
    dim: int = 256
    # 单次 /embeddings 请求的文本条数。DashScope text-embedding-v3 上限是 10，
    # 超了会返回 400 InvalidParameter，所以默认取 10。
    batch_size: int = 10
    timeout: float = 10.0


@dataclass
class LLMConfig:
    url: str = "http://127.0.0.1:8081/v1/chat/completions"
    model: str = "qwen2.5-7b-instruct"
    api_key_env: str = "RAGAGENT_LLM_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 600
    timeout: float = 30.0


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    fusion: str = "rrf"  # rrf | weighted
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    enable_vector: bool = True
    # 向量相似度低于该值不参与融合（哈希向量噪声大时可调高）
    vector_min_score: float = 0.0
    # RRF 各源权重（默认等权；某一路噪声大时可调低）
    bm25_rank_weight: float = 1.0
    vector_rank_weight: float = 1.0
    phrase_rank_weight: float = 1.0
    enable_rerank: bool = True
    rerank_mode: str = "heuristic"  # heuristic | llm | noop
    rerank_top_k: int = 20
    # 融合分低于该值直接拒答（阈值需按评测集校准）
    refuse_threshold: float = 0.0
    # 问题里「知识库没出现过的实词」占比超过该值时拒答；1.0 表示只有全部未见过才拒答
    max_unseen_ratio: float = 1.0
    # 拒答信号里二字词的权重。bigram 多为字符滑窗切出的噪声，默认降权到 0.5
    unseen_bigram_weight: float = 0.5
    # 拒答方式：llm = 用模型判断「仅凭检索到的段落能否回答」（有 Key 时生效）；
    # off = 只用词面信号（离线模式）。词面信号已实测分不开可答/不可答，见 docs/eval-report-example.md
    verify_mode: str = "llm"  # llm | off
    # 给校验器看几条：实测 3 条会漏掉排在第 4 位的答案（可答问题被误拒），5 条更稳
    verify_snippets: int = 5
    # 判定为 need_more 时，扩大到这个窗口再检索一次
    verify_retry_top_k: int = 10
    # 生成后是否做答案接地校验（答案是否被引用段落支持），不支持就降级为摘录
    grounding_mode: str = "llm"  # llm | off


@dataclass
class AgentConfig:
    max_steps: int = 4
    top_k: int = 5
    require_citation: bool = True


@dataclass
class EvalConfig:
    dataset: str = "data/eval/queries.jsonl"
    k_values: tuple[int, ...] = (3, 5)


@dataclass
class AppConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> "AppConfig":
        """从 TOML 加载，缺失项用默认值；支持点号覆盖，例如 {"retrieval.top_k": 3}。"""
        config = cls()
        if path:
            config_path = Path(path)
            if config_path.exists():
                with config_path.open("rb") as fh:
                    raw = tomllib.load(fh)
                config._apply(raw)
        if overrides:
            config._apply(overrides, dotted=True)
        return config

    def _apply(self, raw: Dict[str, Any], dotted: bool = False) -> None:
        if dotted:
            for key, value in raw.items():
                self._set_value(key, value)
            return
        for section, values in raw.items():
            target = getattr(self, section, None)
            if target is None or not isinstance(values, dict):
                continue
            for name, value in values.items():
                if hasattr(target, name):
                    if name == "k_values" and isinstance(value, list):
                        value = tuple(int(v) for v in value)
                    setattr(target, name, value)

    def _set_value(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        if len(parts) != 2:
            raise ValueError(f"覆盖项格式应为 section.field: {dotted_key}")
        section, name = parts
        target = getattr(self, section, None)
        if target is None or not hasattr(target, name):
            raise ValueError(f"未知配置项: {dotted_key}")
        setattr(target, name, value)

    def secret(self, env_name: str) -> str:
        return os.environ.get(env_name, "")

    def llm_api_key(self) -> str:
        return self.secret(self.llm.api_key_env)

    def embedding_api_key(self) -> str:
        return self.secret(self.embedding.api_key_env)

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for section in fields(self):
            value = getattr(self, section.name)
            result[section.name] = {
                f.name: getattr(value, f.name) for f in fields(value)  # type: ignore[arg-type]
            }
        return result


__all__ = [
    "AppConfig", "StorageConfig", "ChunkingConfig", "EmbeddingConfig",
    "LLMConfig", "RetrievalConfig", "AgentConfig", "EvalConfig",
]
