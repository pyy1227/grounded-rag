"""检索编排：三路召回、融合、精排和拒答判定。"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from ..config import AppConfig
from ..embedding import Embedder, build_embedder
from ..llm import ChatModel, build_chat
from ..models import Chunk, SearchHit
from ..storage import Storage
from ..tokenization import content_tokens, normalize, tokenize
from ..verify import ANSWERABLE, CANNOT, build_verifier
from .bm25 import BM25Index
from .fusion import rrf_fuse, weighted_fuse
from .rerank import Reranker, build_reranker
from .vector import VectorIndex

logger = logging.getLogger(__name__)


class Retriever:
    """无状态检索门面；索引在内存中按签名懒加载。"""

    def __init__(
        self,
        storage: Storage,
        config: AppConfig,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        chat: Optional[ChatModel] = None,
    ) -> None:
        self.storage = storage
        self.config = config
        self.embedder = embedder or build_embedder(config)
        self.reranker = reranker or build_reranker(config, chat)

        # 语义拒答校验：有 Key 才启用，离线自动退回词面信号
        self.verifier = None
        if config.retrieval.verify_mode == "llm" and config.llm_api_key():
            self.verifier = build_verifier(
                chat or build_chat(config), snippets=config.retrieval.verify_snippets
            )

        self._signature: Optional[tuple] = None
        self._chunks: List[Chunk] = []
        self._by_id: Dict[str, Chunk] = {}
        self._bm25: Optional[BM25Index] = None
        self._vector: Optional[VectorIndex] = None
        self.last_sources: Dict[str, int] = {}
        self.last_unseen_ratio: float = 0.0
        # 向量那一路是否真的可用；配了 enable_vector 不代表库里有向量
        self.vector_active: bool = False
        self.vector_count: int = 0
        # embedding 服务失败后的冷却时间：不把索引永久置空，到点自动重试
        self._vector_retry_at: float = 0.0
        # 最近一次检索的拒答判定（apply_refusal=False 时也会算，只是不生效）
        self.last_refused: bool = False
        self.last_refusal_reason: Optional[str] = None

    def _current_signature(self) -> tuple:
        return (
            self.storage.count_chunks(),
            len(self.storage.list_documents()),
            self.embedder.model,
        )

    def _ensure_index(self) -> None:
        signature = self._current_signature()
        if signature == self._signature:
            return

        self._chunks = self.storage.iter_chunks(include_parents=False)
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._bm25 = BM25Index(self._chunks)

        vectors = self.storage.load_embeddings(self.embedder.model)
        vectors = {cid: vec for cid, vec in vectors.items() if cid in self._by_id}
        self._vector = VectorIndex.from_embeddings(vectors) if vectors else None
        self.vector_count = len(vectors)
        self.vector_active = self._vector is not None and self.config.retrieval.enable_vector
        # 索引重建（重新入库/换模型）后立即允许重试向量
        self._vector_retry_at = 0.0
        self._warn_if_vectors_missing()
        self._signature = signature

    def index_status(self) -> Dict[str, object]:
        """向量这一路是否可用、加载了几条、是否在失败冷却中。给 /health 和 stats 用。"""
        self._ensure_index()
        return {
            "vector_active": self.vector_active,
            "vectors": self.vector_count,
            "retry_pending": time.monotonic() < self._vector_retry_at,
        }

    def _warn_if_vectors_missing(self) -> None:
        """开了向量检索但库里没有对应模型的向量时告警，不静默降级。

        静默降级时 vector 分数全是 0、reasons 里也没有 vector#，接口照常返回，
        看着像三路召回，其实只跑了两路。
        """
        if not self.config.retrieval.enable_vector or self.vector_active or not self._chunks:
            return
        existing = self.storage.list_embedding_models()
        logger.warning(
            "向量检索已启用，但库里没有 model=%s 的向量（现有：%s），本次只走 BM25+短语。"
            "修法：ragagent --config <config> ingest <docs> --force",
            self.embedder.model,
            existing or "无",
        )

    def invalidate(self) -> None:
        self._signature = None

    # 检索
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        use_vector: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
        apply_refusal: bool = True,
    ) -> List[SearchHit]:
        """检索。

        apply_refusal=False 时只算拒答判定、不清空结果。评测要分开度量检索命中和
        该不该拒答，否则 recall@k 里混进拒答策略，看不出检索本身的变化。
        """
        query = str(query or "").strip()
        if not query:
            return []

        hits = self._retrieve(query, top_k, candidate_k, use_vector, use_rerank)
        if not hits:
            self.last_refused = True
            self.last_refusal_reason = "no_candidates"
            self.last_unseen_ratio = self.unseen_ratio(query)
            return []

        reason, hits = self._refusal_decision(query, hits)
        self.last_refused = reason is not None
        self.last_refusal_reason = reason
        if reason and apply_refusal:
            self.last_sources["refused_by"] = reason
            return []
        return hits

    def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        use_vector: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[SearchHit]:
        """纯检索：三路召回、融合、精排，不做拒答判断。

        单独拆出来是因为 need_more 判定要二次检索，评测也要直接拿纯检索结果算 recall。
        """
        self._ensure_index()
        if not self._chunks:
            return []

        top_k = int(top_k or self.config.retrieval.top_k)
        candidate_k = int(candidate_k or self.config.retrieval.candidate_k)
        use_vector = self.config.retrieval.enable_vector if use_vector is None else use_vector

        bm25_hits = self._bm25.search(query, candidate_k) if self._bm25 else []
        vector_hits = []
        if use_vector and self._vector is not None and time.monotonic() >= self._vector_retry_at:
            try:
                query_vector = self.embedder.embed([query])[0]
                min_score = float(self.config.retrieval.vector_min_score)
                vector_hits = [
                    (chunk_id, score)
                    for chunk_id, score in self._vector.search(query_vector, candidate_k)
                    if score >= min_score
                ]
                self.vector_active = True
            except Exception as exc:  # embedding 服务不可用时降级，不影响主流程
                # 不要在这里把 self._vector 置空：一次网络抖动就会永久退化成
                # BM25+短语，直到重新入库才恢复。这里只进冷却，下次检索继续尝试。
                self.vector_active = False
                self._vector_retry_at = time.monotonic() + 30.0
                self.last_sources = {"vector_error": 1}
                logger.warning("向量检索失败，本次降级为 BM25+短语（30 秒后重试）：%s", exc)
        phrase_hits = self._phrase_search(query, candidate_k)

        rank_lists: Dict[str, List[str]] = {
            "bm25": [hit.chunk.chunk_id for hit in bm25_hits],
            "vector": [chunk_id for chunk_id, _ in vector_hits],
            "phrase": [chunk_id for chunk_id, _ in phrase_hits],
        }
        self.last_sources = {source: len(ids) for source, ids in rank_lists.items()}

        if self.config.retrieval.fusion == "weighted":
            score_lists = {
                "bm25": {hit.chunk.chunk_id: hit.score for hit in bm25_hits},
                "vector": {chunk_id: score for chunk_id, score in vector_hits},
                "phrase": {chunk_id: score for chunk_id, score in phrase_hits},
            }
            weights = {
                "bm25": self.config.retrieval.bm25_weight,
                "vector": self.config.retrieval.vector_weight,
                "phrase": 0.1,
            }
            fused = weighted_fuse(score_lists, weights)
        else:
            fused = rrf_fuse(
                rank_lists,
                k=self.config.retrieval.rrf_k,
                weights={
                    "bm25": self.config.retrieval.bm25_rank_weight,
                    "vector": self.config.retrieval.vector_rank_weight,
                    "phrase": self.config.retrieval.phrase_rank_weight,
                },
            )

        if not fused:
            return []

        bm25_score = {hit.chunk.chunk_id: hit.score for hit in bm25_hits}
        vector_score = {cid: score for cid, score in vector_hits}
        bm25_rank = {hit.chunk.chunk_id: index for index, hit in enumerate(bm25_hits, 1)}
        vector_rank = {cid: index for index, (cid, _) in enumerate(vector_hits, 1)}

        hits: List[SearchHit] = []
        for chunk_id, fused_score in fused.items():
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue
            hit = SearchHit(
                chunk=chunk,
                score=fused_score,
                fused_score=fused_score,
                bm25_score=bm25_score.get(chunk_id, 0.0),
                vector_score=vector_score.get(chunk_id, 0.0),
                bm25_rank=bm25_rank.get(chunk_id),
                vector_rank=vector_rank.get(chunk_id),
            )
            if hit.bm25_rank:
                hit.reasons.append(f"bm25#{hit.bm25_rank}")
            if hit.vector_rank:
                hit.reasons.append(f"vector#{hit.vector_rank}")
            hits.append(hit)

        hits.sort(key=lambda item: item.fused_score, reverse=True)
        hits = hits[: self.config.retrieval.rerank_top_k]

        use_rerank = self.config.retrieval.enable_rerank if use_rerank is None else use_rerank
        if use_rerank and self.reranker is not None:
            hits = self.reranker.rerank(query, hits, top_k)
        else:
            hits = hits[:top_k]
            for hit in hits:
                hit.score = hit.fused_score

        return hits

    def _refusal_decision(self, query: str, hits: List[SearchHit]) -> Tuple[Optional[str], List[SearchHit]]:
        """判断该不该拒答，返回（原因, 最终结果）；不该拒答时原因为 None。

        结果清不清空由调用方决定，评测要分别看判定结果和检索结果。
        need_more 时扩大窗口二次检索，只有确实没有才拒答。
        """
        threshold = float(self.config.retrieval.refuse_threshold or 0.0)
        if threshold > 0 and (not hits or hits[0].score < threshold):
            return "score_threshold", hits

        unseen_ratio = self.unseen_ratio(query)
        self.last_unseen_ratio = unseen_ratio

        # 语义校验优先；词面信号实测分不开可答和不可答
        if self.verifier is not None:
            verdict, _reason = self.verifier.verify_tiered(query, hits)
            if verdict == ANSWERABLE:
                return None, hits
            if verdict == CANNOT:
                return "verifier_cannot", hits

            # need_more：资料相关但可能没检索全，扩大窗口再查一次
            retry_k = int(self.config.retrieval.verify_retry_top_k)
            if retry_k <= len(hits):
                retry_k = len(hits) + 5
            wider = self._retrieve(query, top_k=retry_k)
            if wider and len(wider) > len(hits):
                verdict, _reason = self.verifier.verify_tiered(query, wider)
                if verdict == ANSWERABLE:
                    self.last_sources["retried"] = 1
                    return None, wider
                if verdict == CANNOT:
                    return "verifier_cannot", wider
            self.last_sources["retried"] = 1
            return "verifier_need_more", wider or hits

        max_unseen = float(self.config.retrieval.max_unseen_ratio)
        if max_unseen < 1.0 and unseen_ratio >= max_unseen:
            return "unseen_ratio", hits
        if max_unseen >= 1.0 and unseen_ratio >= 1.0:
            return "all_terms_unseen", hits
        return None, hits

    def unseen_ratio(self, query: str) -> float:
        """问题里的实词有多大比例没在库里出现过。

        只统计实词：虚词和语气助词（吗/了/能/我）跟可答性无关，等权计数会把
        口语化问法的比例推到 0.8 以上，造成大量误拒。
        二字词按 unseen_bigram_weight（默认 0.5）降权，它们多是字符滑窗切出的噪声。
        """
        if self._bm25 is None:
            return 0.0
        tokens = set(content_tokens(query))
        if not tokens:
            return 0.0
        bigram_weight = float(self.config.retrieval.unseen_bigram_weight)
        total = 0.0
        unseen = 0.0
        for token in tokens:
            weight = bigram_weight if len(token) >= 2 else 1.0
            total += weight
            if self._bm25.df(token) == 0:
                unseen += weight
        return unseen / total if total else 0.0

    # 短语召回（第三路）
    def _phrase_search(self, query: str, top_k: int) -> List[tuple[str, float]]:
        """只做精确子串匹配（别名、标准问法）。

        这一路只用原文命中，语义泛化交给 BM25 和向量，避免把覆盖率噪声带进融合。
        """
        query_norm = normalize(query)
        if len(query_norm) < 2:
            return []

        scores: List[tuple[str, float]] = []
        for chunk in self._chunks:
            if query_norm in normalize(chunk.text):
                scores.append((chunk.chunk_id, 2.0))
        return scores[:top_k]

    def expand(self, hit: SearchHit, max_chars: int = 1200) -> str:
        """子块命中后回填父块，兼顾精度和上下文完整。"""
        if hit.chunk.parent_id:
            parent = self.storage.get_chunk(hit.chunk.parent_id)
            if parent is not None:
                return parent.text[:max_chars]
        return hit.chunk.text[:max_chars]


__all__ = ["Retriever"]
