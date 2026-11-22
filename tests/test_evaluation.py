"""评测指标测试：用桩检索器验证数学正确，不依赖真实索引。"""
from __future__ import annotations

import json

from ragagent.config import AppConfig
from ragagent.evaluation import evaluate, load_dataset, sweep_thresholds
from ragagent.models import Chunk, EvalCase, SearchHit


def _hit(doc_id: str, section: str, text: str = "内容") -> SearchHit:
    chunk = Chunk(
        chunk_id=f"{doc_id}-{section}",
        doc_id=doc_id,
        ordinal=0,
        section_path=section,
        kind="prose",
        text=text,
    )
    return SearchHit(chunk=chunk, score=1.0, fused_score=1.0)


class StubRetriever:
    def __init__(self) -> None:
        self.config = AppConfig()
        self.last_unseen_ratio = 0.0
        self._hits = {
            "q1": [_hit("售后政策", "退货运费")],
            "q2": [_hit("其他文档", "无关章节"), _hit("售后政策", "退款时效")],
            "q3": [],
        }

    def search(self, query, top_k=None, use_rerank=None):
        return self._hits.get(query, [])


def test_metrics_math():
    retriever = StubRetriever()
    cases = [
        EvalCase(query="q1", gold=["售后政策::退货运费"]),
        EvalCase(query="q2", gold=["售后政策::退款时效"]),
        EvalCase(query="q3", gold=[], answerable=False),
    ]
    result = evaluate(retriever, cases, k_values=(1, 3))
    metrics = result["metrics"]
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["over_refusal"] == 0.0
    assert metrics["refusal_accuracy"] == 1.0


def test_dataset_loader(workdir):
    path = workdir / "eval.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "gold": ["d::s"], "answerable": True}, ensure_ascii=False),
                json.dumps({"query": "q2", "gold": [], "answerable": False}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert len(cases) == 2
    assert cases[0].gold == ["d::s"]
    assert cases[1].answerable is False


def test_sweep_restores_config():
    retriever = StubRetriever()
    retriever.config.retrieval.max_unseen_ratio = 0.8
    cases = [EvalCase(query="q1", gold=["售后政策::退货运费"])]
    rows = sweep_thresholds(retriever, cases, thresholds=(1.0, 0.5))
    assert len(rows) == 2
    assert retriever.config.retrieval.max_unseen_ratio == 0.8
