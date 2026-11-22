"""离线评测：检索质量与拒答质量分开度量。

指标定义：
- recall@k：answerable 样本中，检索结果前 k 条里至少命中一条 gold 的比例（不经过拒答过滤）
- recall@k(effective)：在 recall@k 基础上再叠加拒答——被拒答的样本算未命中
- MRR：第一条 gold 命中排名的倒数均值（不经过拒答过滤）
- refusal_accuracy：不可回答样本被正确拒答的比例
- over_refusal：可回答样本被误拒的比例（越低越好）
- 阈值扫描：max_unseen_ratio 从小到大的取舍，用 recall@k(effective) 看业务侧真实损失

为什么要拆：如果 recall@k 直接在「拒答之后」的结果上算，检索变差和拒答变激进会混成
同一个数字，既看不出是哪一层的问题，也没法比较两路召回的差异。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import AppConfig
from .models import CaseResult, EvalCase, SearchHit
from .retrieval.retriever import Retriever
from .storage import Storage


def load_dataset(path: str | Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            cases.append(
                EvalCase(
                    query=raw["query"],
                    gold=list(raw.get("gold") or []),
                    expected_keywords=list(raw.get("expected_keywords") or []),
                    answerable=bool(raw.get("answerable", True)),
                )
            )
    return cases


def evaluate(
    retriever: Retriever,
    cases: Iterable[EvalCase],
    k_values: Iterable[int] = (3, 5),
    refuse_threshold: Optional[float] = None,
    max_unseen_ratio: Optional[float] = None,
    dataset: str = "",
) -> Dict[str, Any]:
    cases = list(cases)
    k_values = sorted({int(k) for k in k_values})
    max_k = max(k_values)

    config = retriever.config
    original_threshold = config.retrieval.refuse_threshold
    original_unseen = config.retrieval.max_unseen_ratio
    if refuse_threshold is not None:
        config.retrieval.refuse_threshold = float(refuse_threshold)
    if max_unseen_ratio is not None:
        config.retrieval.max_unseen_ratio = float(max_unseen_ratio)

    results: List[CaseResult] = []
    unseen_ratios: List[float] = []
    try:
        for case in cases:
            # 关掉拒答过滤，先把「检索命中了没有」单独拿到手
            hits: List[SearchHit] = retriever.search(
                case.query, top_k=max_k, use_rerank=True, apply_refusal=False
            )
            refused = bool(getattr(retriever, "last_refused", False)) or not hits
            unseen_ratios.append(retriever.last_unseen_ratio)
            ranks = []
            keys = []
            for rank, hit in enumerate(hits, start=1):
                hit_key = f"{hit.chunk.doc_id}::{hit.chunk.section_path}"
                keys.append(hit_key)
                for gold in case.gold:
                    if _match(hit, gold):
                        ranks.append(rank)
                        break
            results.append(
                CaseResult(
                    case=case,
                    hit_keys=keys,
                    ranks=ranks,
                    refused=refused,
                )
            )
    finally:
        config.retrieval.refuse_threshold = original_threshold
        config.retrieval.max_unseen_ratio = original_unseen

    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    answerable_ratios = [ratio for ratio, r in zip(unseen_ratios, results) if r.case.answerable]
    unanswerable_ratios = [ratio for ratio, r in zip(unseen_ratios, results) if not r.case.answerable]

    metrics: Dict[str, Any] = {
        "dataset": dataset,
        "cases": len(cases),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mean_unseen_ratio_answerable": round(
            sum(answerable_ratios) / max(1, len(answerable_ratios)), 4
        ),
        "mean_unseen_ratio_unanswerable": round(
            sum(unanswerable_ratios) / max(1, len(unanswerable_ratios)), 4
        ) if unanswerable_ratios else None,
    }
    for k in k_values:
        hit_count = sum(1 for r in answerable if r.ranks and min(r.ranks) <= k)
        metrics[f"recall@{k}"] = round(hit_count / max(1, len(answerable)), 4)
        # 叠加拒答后的有效召回：业务上用户真正「能拿到答案」的比例
        effective = sum(
            1 for r in answerable if not r.refused and r.ranks and min(r.ranks) <= k
        )
        metrics[f"recall@{k}_effective"] = round(effective / max(1, len(answerable)), 4)

    metrics["mrr"] = round(
        sum(1.0 / min(r.ranks) for r in answerable if r.ranks) / max(1, len(answerable)),
        4,
    )
    metrics["over_refusal"] = round(
        sum(1 for r in answerable if r.refused) / max(1, len(answerable)), 4
    )
    if unanswerable:
        metrics["refusal_accuracy"] = round(
            sum(1 for r in unanswerable if r.refused) / len(unanswerable), 4
        )
    else:
        metrics["refusal_accuracy"] = None

    for k in k_values:
        metrics[f"cases_with_gold_missing@{k}"] = [
            {"query": r.case.query, "gold": r.case.gold}
            for r in answerable
            if not (r.ranks and min(r.ranks) <= k)
        ][:10]

    return {
        "metrics": metrics,
        "cases": [
            {
                "query": r.case.query,
                "gold": r.case.gold,
                "refused": r.refused,
                "first_gold_rank": min(r.ranks) if r.ranks else None,
                "top_hits": r.hit_keys[:5],
            }
            for r in results
        ],
        "config": config.as_dict(),
    }


def sweep_thresholds(
    retriever: Retriever,
    cases: Iterable[EvalCase],
    thresholds: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> List[Dict[str, Any]]:
    """拒答阈值扫描：max_unseen_ratio 越小越激进（更多拒答）。

    输出 recall@5(effective) / 误拒率 / 拒答准确率三者的取舍表，用于按业务风险选点。
    recall 用 effective 版本——用户视角的损失里，被误拒和检索没命中一样算没拿到答案。
    """
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        result = evaluate(retriever, cases, k_values=(5,), max_unseen_ratio=threshold)
        metrics = result["metrics"]
        rows.append(
            {
                "threshold": threshold,
                "recall@5": metrics.get("recall@5_effective"),
                "recall@5_retrieval_only": metrics.get("recall@5"),
                "over_refusal": metrics.get("over_refusal"),
                "refusal_accuracy": metrics.get("refusal_accuracy"),
            }
        )
    return rows


def write_report(result: Dict[str, Any], path: str | Path, extra: Optional[Dict[str, Any]] = None) -> str:
    metrics = result["metrics"]
    lines = [
        "# RAG 评测报告",
        "",
        f"- 数据集：`{metrics.get('dataset') or '-'}`",
        f"- 生成时间：{metrics.get('generated_at')}",
        f"- 样本：{metrics.get('cases')}（可回答 {metrics.get('answerable')}，不可回答 {metrics.get('unanswerable')}）",
        "",
        "## 核心指标",
        "",
        "`recall@k` 只衡量检索命中（不经过拒答）；`recall@k_effective` 叠加拒答，"
        "两者之差就是拒答策略的代价。",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key in ("recall@3", "recall@3_effective", "recall@5", "recall@5_effective",
                "mrr", "over_refusal", "refusal_accuracy",
                "mean_unseen_ratio_answerable", "mean_unseen_ratio_unanswerable"):
        if key in metrics:
            value = metrics[key]
            lines.append(f"| {key} | {'-' if value is None else value} |")

    if extra and extra.get("sweep"):
        lines += [
            "",
            "## 拒答阈值扫描（max_unseen_ratio）",
            "",
            "阈值越小越激进：越容易把「没见过的问题」判为拒答。",
            "",
            "| max_unseen_ratio | recall@5 | 误拒率 | 拒答准确率 |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for row in extra["sweep"]:
            lines.append(
                f"| {row['threshold']} | {row['recall@5']} | {row['over_refusal']} | {row['refusal_accuracy']} |"
            )

    retrieval = result.get("config", {}).get("retrieval", {})
    lines += [
        "",
        "## 检索配置",
        "",
        "```json",
        json.dumps(retrieval, ensure_ascii=False, indent=2),
        "```",
    ]

    missing = metrics.get("cases_with_gold_missing@5") or []
    if missing:
        lines += ["", "## 未命中样例（前 10）", ""]
        for item in missing:
            lines.append(f"- `{item['query']}` → 期望 {item['gold']}")

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def _match(hit: SearchHit, gold: str) -> bool:
    """gold 形如 `doc_id::章节关键字`；章节关键字为空时只比 doc_id。

    只认 `section_path`，**不看正文**。宽松口径（章节名 or 正文含关键字）会把
    「见退货运费表格」这类只是引用别处的段落也算命中，机制上会虚高；
    本集实测两种口径数值相同（0.963 / 0.778），因为 gold 都写成了真实章节名，
    但保留严格口径是为了避免以后新增题目时再踩这个坑。
    """
    if "::" in gold:
        doc_id, keyword = gold.split("::", 1)
    else:
        doc_id, keyword = gold, ""
    if hit.chunk.doc_id != doc_id:
        return False
    if not keyword:
        return True
    return keyword in hit.chunk.section_path


__all__ = ["load_dataset", "evaluate", "sweep_thresholds", "write_report"]
