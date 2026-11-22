# 外部数据集验证

内部那套 57 条评测集（dev 34 + test 23）是自己出题、自己标 gold、还在同一份上选阈值——只能做内部对比。
这份文档记录**用公开数据集**做的外部验证：题不是我们写的，用来回答「你的结论是不是自己给自己打分」。

## 数据来源

| 项 | 值 |
| --- | --- |
| 数据集 | [`C-MTEB/T2Reranking`](https://huggingface.co/datasets/C-MTEB/T2Reranking)（中文，6,129 条 query / positive / hard negative 三元组） |
| 获取方式 | 经 `hf-mirror.com` 镜像下载 parquet |
| 任务改造 | 原本是 reranking，这里把正例与难负例组织成小语料，转成 query→passage 检索任务 |
| 抽样 | 200 条查询；语料 3,000 条（200 条正例 + 2,800 条从 97,269 条去重 passage 里随机补齐），`seed=42` |
| 预处理 | 去掉 `<br>/<img>` 等标签；单条 passage 截断到 1,500 字（原数据最长 6.8 万字，超 embedding 输入上限） |

为什么不用 `C-MTEB/T2Retrieval`、`DuRetrieval`：这两个子集在 HF 上只放了 `corpus` 与 `queries` 两个 parquet，
**没有 qrels**，无法算 recall；`T2Reranking` 把标注内联在数据里，可以直接用。

## 结果

同一份外部语料、同一套检索参数，只换 embedding：

| 指标 | 哈希向量（`config.offline.toml`） | text-embedding-v3（`config.toml`） | 差异 |
| --- | ---: | ---: | ---: |
| recall@3 | 0.6450 | **0.8450** | +20.0pp |
| recall@5 | 0.7050 | **0.9000** | +19.5pp |
| MRR | 0.5657 | **0.7205** | +0.155 |
| 误拒率 | 0.0 | 0.0 | 0 |

## 结论

1. **方向与内部集一致**：真实语义向量显著优于哈希基线（+20pp recall@3）。
   内部集上量到的是 +21.7pp（0.6957 → 0.9130），两份数据同向、量级也相当，
   说明内部集的结论不是「自己出题自己打分」刷出来的。
2. **量级更可信**：外部集的题是真实用户问法（`a7s2和a7m3视频谁强`、`剃完胡子红红的怎么消`），
   语料里有 97k 条候选来源、且混着难负例，不是 15 篇文档里挑 121 个 chunk 能比的。
3. **拒答仍未覆盖**：这个外部集全部是「可回答」样本，所以只能验证检索层，
   拒答的缺陷（见 [eval-report-example.md](eval-report-example.md)）依旧成立、未被外部数据洗白。

## 局限（写清楚，别当成线上指标）

- **小语料抽样**：3,000 条语料里找 1 条正例，比在百万级真实语料里容易，recall 绝对值偏乐观；
- **难负例分布**：语料里混入了同题材的难负例，难度高于随机采样，与真实流量分布也不同；
- **只测 200 条查询**：1 条题 = 0.5pp，别拿小数点后两位说事；
- **不测生成质量**：只度量检索命中，不评价最终答案对不对。

## 复现

```bash
# 1. 生成外部评测集
#    注意：pyarrow 17+ 要求 numpy>=2，本项目锁了 numpy<2，所以要装 16.x
pip install "pyarrow<17"
python scripts/prepare_cmteb.py --dataset T2Reranking --queries 200 --corpus 3000

# 2. 入库（两条路各自一个库）
ragagent --config config.offline.toml --db data/external/T2Reranking/hash.db \
         ingest-jsonl data/external/T2Reranking/passages.jsonl
ragagent --config config.toml --db data/external/T2Reranking/real.db \
         ingest-jsonl data/external/T2Reranking/passages.jsonl

# 3. 评测
ragagent --config config.offline.toml --db data/external/T2Reranking/hash.db \
         eval --dataset data/external/T2Reranking/queries.jsonl --no-sweep --report reports/ext-hash.md
ragagent --config config.toml --db data/external/T2Reranking/real.db \
         eval --dataset data/external/T2Reranking/queries.jsonl --no-sweep --report reports/ext-real.md
```

`--no-sweep` 是必要的：阈值扫描会把 6 × 200 条查询各跑一遍，额外 1,200 次 embedding 调用。
