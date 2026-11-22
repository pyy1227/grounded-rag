# RAG 评测报告（示例）

> 由 `ragagent eval --report reports/eval.md` 生成后整理。
> 数据为仓库自带 32 条演示评测集，**用于对比配置差异，不代表线上效果**。

- 数据集：`data/eval/queries.jsonl`
- 样本：32（可回答 27，不可回答 5）
- 运行模式：BM25 + text-embedding-v3（1024 维）+ 短语召回 → RRF 融合 → heuristic 精排
- 分词：bigram

## 核心指标

| 指标 | 数值 |
| --- | ---: |
| recall@3 | 1.0 |
| recall@5 | 1.0 |
| MRR | 0.8827 |
| 误拒率（over_refusal） | 0.0 |
| 拒答准确率 | 0.8 |
| 可回答问题的平均未见词比例 | 0.2463 |
| 不可回答问题的平均未见词比例 | 0.8417 |

## 拒答阈值扫描（max_unseen_ratio）

阈值越小越激进：越容易把「知识库没见过的问题」判为拒答。

| max_unseen_ratio | recall@5 | 误拒率 | 拒答准确率 |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 0.0 | 0.4 |
| 0.9 | 1.0 | 0.0 | 0.4 |
| **0.8（当前）** | **1.0** | **0.0** | **0.8** |
| 0.7 | 1.0 | 0.0 | 0.8 |
| 0.6 | 0.963 | 0.037 | 0.8 |
| 0.5 | 0.7778 | 0.2222 | 1.0 |

## 结论与取舍

- 选 `max_unseen_ratio = 0.8`：拒答准确率 0.8，误拒率 0；5 条不可回答里拦住 4 条，可回答问题一条没误伤。
- 降到 0.5 可把拒答准确率提到 1.0，但 recall@5 掉到 0.78、误拒率 22%——客服场景不可接受。
- 对比哈希向量（`config.offline.toml`）：recall@3/5 同为 1.0，MRR 0.8765 → 0.8827 基本打平；
  阈值取舍点完全一致，因为拒答信号来自 BM25 词表覆盖，与向量质量无关。
- **这套 32 条集 recall 已饱和**，只够验证评测闭环和阈值取舍，不足以比较 embedding 或召回策略；
  要证明某一路有用，需要能拉开差距的集（同义改写、多跳、长尾问法）。

## 复现方式

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY
ragagent ingest data/docs --force
ragagent eval --dataset data/eval/queries.jsonl --report reports/eval.md
```

没有 API Key 时用离线配置，指标会变成 recall@3 1.0 / MRR 0.8765：

```bash
ragagent --config config.offline.toml ingest data/docs --force
ragagent --config config.offline.toml eval --report reports/eval-offline.md
```
