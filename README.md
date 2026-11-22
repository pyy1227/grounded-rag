# 企业知识库 RAG 助手（可评测 + Agent 层）

一个**可度量**的 RAG 系统：文档入库 → 结构化切分 → 三路混合召回 → 融合 → 精排 → 带引用回答 →
离线评测（recall@k / MRR / 拒答准确率），并在上层提供工具调用型 Agent。

> 重点不是"能问答"，而是"能证明检索变好了"和"没依据时敢拒答"。

## 架构

```
                 ┌─────────────────────────────────────────────┐
  文档(md/pdf/    │ 解析 → 结构化切分(FAQ/表格/父子块) → Embedding │
  docx/xlsx) ───▶ │            → SQLite(chunks + 向量)           │
                 └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
             BM25 召回                        向量召回 / 短语召回
                    └──────────── RRF 融合 ────────────┘
                                      ▼
                             Heuristic / LLM 精排
                                      ▼
                          阈值拒答 → 带引用回答([n])
                                      ▼
                     ┌────────────────┴────────────────┐
                     │            Agent 层              │
                     │ search / open_chunk / list_docs  │
                     │ compare / submit_answer          │
                     │ 多步循环 + trace + 引用校验      │
                     └─────────────────────────────────┘
```

## 快速开始

```bash
cd rag-agent
uv sync                      # 或 pip install -e ".[dev]"

# 仓库带两份配置：
#   config.toml         → DeepSeek(对话) + DashScope(向量)，需要 .env 里的 key
#   config.offline.toml → 哈希向量 + 规则替身，零 Key 可跑
cp .env.example .env         # 然后填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY

# 1. 入库（自带 demo 文档）
ragagent ingest data/docs

# 2. 检索 / 回答
ragagent search "退货运费谁承担" -k 5
ragagent ask "退货运费谁承担"
ragagent agent "退货时效和换货时效有什么区别"

# 3. 评测（产出报告）
ragagent eval --report reports/eval.md

# 4. HTTP 服务
ragagent serve --port 8000
# 浏览器打开 http://127.0.0.1:8000/ 就是网页版
# curl -X POST localhost:8000/agent -H "Content-Type: application/json" \
#      -d '{"query":"发票什么时候能开"}'
```

`.env` 由 python-dotenv 自动加载（已存在的环境变量优先）。填好 key 后 `ask` 会走
**DeepSeek + text-embedding-v3**，返回 `mode=llm`；不配 key 时用离线配置跑全流程：

```bash
ragagent --config config.offline.toml ingest data/docs
ragagent --config config.offline.toml ask "退货运费谁承担"   # mode=extractive
```

## 演示

```powershell
# 离线版：哈希向量 + 规则替身，不需要任何 Key、不联网
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Offline -Pause -FullEval

# 真实模型版：DeepSeek 回答 + DashScope 向量（需要 .env）
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Pause
```

脚本按七幕顺序跑：索引现状 → 入库幂等 → 检索四层分数 → 带引用回答 → 拒答 → Agent trace → 评测报告。
每一幕的预期输出和讲解词见 [docs/demo.md](docs/demo.md)。

## 网页界面

`ragagent serve` 起来后打开 <http://127.0.0.1:8000/>，同源提供四个视图，**零构建、零外部依赖**：

| 视图 | 看什么 |
| --- | --- |
| 问答 | 答案 + 可点开的 `[n]` 引用原文、是否拒答、`mode`（llm / extractive / agent）、耗时 |
| 检索 | 每条命中的 `bm25 / vector / fused / rerank` 四层分数条，解释「为什么它排在前面」 |
| 知识库 | 已入库文档、chunk 数、版本、入库时间，并可触发一次入库（幂等演示） |
| 评测 | recall@3/5、MRR、拒答准确率、误拒率、阈值扫描表、逐条样本 |

支持深链接，演示时可以直接发链接：

```text
http://127.0.0.1:8000/?q=退货运费谁承担
http://127.0.0.1:8000/?q=退货时效和换货时效有什么区别&mode=agent
http://127.0.0.1:8000/?s=退货运费谁承担
```

前端只有一个 `src/ragagent/static/index.html`（原生 JS + fetch，无 npm、无打包），
由 FastAPI 挂在根路径；后端接口仍是纯 JSON，CLI 用法完全不变。

## 配置要点

| 配置 | 说明 |
| --- | --- |
| `retrieval.fusion` | `rrf`（默认，只看排名）或 `weighted`（可调权重） |
| `retrieval.rerank_mode` | `heuristic` / `llm` / `noop`，用于消融对比 |
| `retrieval.refuse_threshold` | 精排分低于该值直接拒答（次要信号） |
| `retrieval.max_unseen_ratio` | 问题里「知识库没出现过的实词」占比阈值，越小越容易拒答（主要信号，用 `eval` 扫描校准） |
| `chunking.max_chars` / `overlap_chars` | 切分粒度，影响召回和上下文完整度 |
| `embedding.provider` | `hash`（离线）或 `remote`（OpenAI 兼容 /embeddings） |
| `embedding.batch_size` | 单次 embed 请求的条数，默认 10（DashScope text-embedding-v3 上限，超了 400） |
| `api_key_env` | key 从哪个环境变量读。`config.toml` 指向 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` |
| `llm.*` | 回答/Agent 用的模型；**不要用 `deepseek-reasoner`**，它不支持 function calling |

## 评测（实测数字）

```bash
ragagent eval --dataset data/eval/queries.jsonl --report reports/eval.md
```

仓库自带 32 条演示评测集（27 条可回答 + 5 条不可回答）。同一份评测集、同一套检索参数，
两种向量实现的实测对比：

| 指标 | 哈希向量（`config.offline.toml`） | text-embedding-v3（`config.toml`） |
| --- | ---: | ---: |
| recall@3 | 1.0 | 1.0 |
| recall@5 | 1.0 | 1.0 |
| MRR | 0.8765 | **0.8827** |
| 误拒率 | 0.0 | 0.0 |
| 拒答准确率 | 0.8 | 0.8 |

两套向量在这套 32 条集上已经接近打平（MRR 0.8765 vs 0.8827）：recall@3/5 双双饱和在 1.0，
说明这套集区分不出 embedding 质量，真实语义向量的收益要换到同义改写、长尾问法上才看得出来。
`max_unseen_ratio` 的最优取舍点两边完全一致——拒答信号来自 BM25 词表，与向量无关，
侧面说明两路召回是互补的，不是互相替代。

完整报告见 [docs/eval-report-example.md](docs/eval-report-example.md)，包含阈值扫描的取舍表：
`max_unseen_ratio` 从 1.0 降到 0.5，拒答准确率 0.4 → 1.0，但 recall@5 从 1.0 掉到 0.78、误拒率升到 22%。

> 注意：这是 32 条样本的对比实验，样本小且 recall 已饱和，数字只用于验证「评测闭环能跑通」
> 和「阈值取舍有依据」，**不能**用来论证某一路召回/某个 embedding 更好，也不代表线上精度。
> 换 embedding 模型后必须 `ragagent ingest data/docs --force` 重建索引，否则新旧向量不在同一空间。
> 判断向量到底有没有生效看 `ragagent stats` 的 `vector_active` / `vectors`——**不要只看 `embedder` 名字**，
> 配置写了 `enable_vector = true` 不等于库里真有向量（缺向量会打 WARNING 并降级为 BM25+短语）。
> 想换分词器：`pip install -e ".[zh]"` 后设 `RAGAGENT_TOKENIZER=jieba`（换分词器后同样需重新 ingest）。

报告包含：

- `recall@3` / `recall@5`：gold 是否进入前 k
- `MRR`：首个命中排名的倒数均值
- `refusal_accuracy`：不可回答问题的正确拒答率
- `over_refusal`：可回答问题被误拒的比例
- **阈值扫描**：`refuse_threshold` 从 0 到 0.6 的取舍表

评测集格式（`data/eval/queries.jsonl`）：

```json
{"query":"退货运费谁承担","gold":["售后政策::运费"],"answerable":true}
{"query":"支持货到付款吗","gold":[],"answerable":false}
```

## 目录结构

```
rag-agent/
├── src/ragagent/
│   ├── parsing.py        # md/pdf/docx/xlsx 解析
│   ├── chunking.py       # FAQ/表格/父子块切分
│   ├── indexer.py        # 增量入库（sha256 幂等）
│   ├── storage.py        # SQLite（chunks + 向量 + 评测记录）
│   ├── embedding.py      # remote / hash 两种 provider
│   ├── retrieval/
│   │   ├── bm25.py       # 自写 BM25
│   │   ├── vector.py     # 余弦检索
│   │   ├── fusion.py     # RRF / 加权
│   │   ├── rerank.py     # 启发式 / LLM 精排
│   │   └── retriever.py  # 三路召回编排
│   ├── generation.py     # 带引用回答 + 拒答
│   ├── evaluation.py     # 指标 + 阈值扫描 + 报告
│   ├── agent/            # 工具调用 Agent（多步 + trace）
│   ├── api.py            # FastAPI
│   ├── cli.py            # 命令行
│   └── static/index.html # 网页（单文件原生 JS，无构建）
├── data/docs/            # demo 文档
├── data/eval/            # 评测集
├── docs/eval-report-example.md  # 评测报告示例
└── tests/                # pytest
```

## 设计取舍（面试可直接讲）

1. **不用 LangChain**：切分、融合、精排、评测都需要可解释的中间分数，框架会把这些藏起来。
2. **不用向量数据库**：万级 chunk 内存算余弦足够；接口留了 `VectorIndex`，换 pgvector 只改一个文件。
3. **RRF 而不是加权**：三路分数量纲不同，RRF 只看排名，鲁棒；加权只在明确调过参时用。
4. **拒答是合法动作**：用「未见实词比例」做主要信号，阈值由评测集的 recall/误拒取舍决定，宁可说不知道也不编。
5. **父子块**：子块保证召回精度，父块保证回答上下文完整。
6. **Agent 只给工具出口**：检索/读全文/对比/提交答案都是工具，工具之外不能产生副作用。
7. **网页只用一个静态文件**：演示要看的是「引用、分数、trace」，不是 SPA 工程；不引入 Node 构建链，
   页面和 API 同源，`serve` 一条命令就能跑，也不影响 CLI 和接口。

## 测试

```bash
pytest -q
```

覆盖：切分（FAQ/表格/长文）、BM25 排序、RRF 融合、评测指标、Agent 多步循环（FakeChat 注入）、
静态页挂载（能打开且不挡 API 路由）、向量状态可见性（缺向量时告警而不是静默降级）。

## 实现规模

- 33 个 Python 文件、约 3,170 行代码（其中测试 8 个文件约 380 行 / 28 个用例），外加一个约 1,120 行的静态页
- 纯 Python 后端 + 一个 HTML 文件；无 Node 构建链、无外部数据库服务：SQLite 单文件即可跑

## 已知限制

- `config.offline.toml` 的哈希向量没有语义能力，只用于零依赖跑通；`config.toml` 已接真实 embedding（text-embedding-v3）；
- **评测集只有 32 条且 recall 已饱和**，只够验证评测闭环与阈值取舍，不足以比较 embedding / 召回策略，更不代表线上精度；
- SQLite 单机，多租户/高并发需换 PostgreSQL + pgvector；
- 未做鉴权、限流、重试与结构化日志，属演示级服务，别当生产网关用。
