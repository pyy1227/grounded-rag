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

仓库自带 32 条演示评测集（27 条可回答 + 5 条不可回答）。题面是**口语化改写**，不照抄知识库原词
（例如「是他们发错颜色了，寄回去的快递钱还要我掏吗」），这样才能把「字面匹配」和「语义检索」区分开。

评测把**检索**和**拒答**分开度量：`recall@k` 只算检索命中，`recall@k_effective` 再叠加拒答过滤。

**检索层**（同一份集、同一套检索参数，只换 embedding）：

| 指标 | 哈希向量（`config.offline.toml`） | text-embedding-v3（`config.toml`） |
| --- | ---: | ---: |
| recall@3 | 0.7778 | **0.9630** |
| recall@5 | 0.8148 | **0.9630** |
| MRR | 0.6142 | **0.8457** |

哈希向量本质是字符 bigram 的袋，口语改写一上就掉；语义向量能靠 paraphrase 命中，差 15~23 个百分点。

gold 只认 `section_path`，**不认正文**：早期版本写成「章节名 or 正文含关键字」，会让「见退货运费表格」
这类只是引用别处的段落也算命中，机制上注定虚高。

**拒答层**：两套向量完全一样（误拒率 0.4815、拒答准确率 0.6）——拒答信号来自 BM25 词表，与向量无关。
而且这个信号在本集上**同时错了两头**：可回答问题的平均未见词比例 0.6643、不可回答问题 0.7774，
两个分布几乎重叠，扫遍 1.0~0.5 都找不到能同时兼顾的阈值。

> 旧集（照抄知识库原词时）这两个数是 0.2463 / 0.8417，所以 0.8 那个点看起来又准又不误伤——
> 那是**题面复用原词**带来的虚高。这是本项目目前最明确的已知缺陷：拒答需要换信号
> （例如用检索覆盖度 + 模型判断，而不是数未见词），不是调阈值能解决的。

完整报告见 [docs/eval-report-example.md](docs/eval-report-example.md)，包含阈值扫描的取舍表和根因分析。

报告包含：

- `recall@3` / `recall@5`：gold 是否进入前 k（**不经过拒答**，纯检索）
- `recall@3_effective` / `recall@5_effective`：叠加拒答后的有效召回，两者之差即拒答的代价
- `MRR`：首个命中排名的倒数均值
- `refusal_accuracy`：不可回答问题的正确拒答率
- `over_refusal`：可回答问题被误拒的比例
- **阈值扫描**：`max_unseen_ratio` 从 1.0 到 0.5 的取舍表（用 `recall@5_effective` 看业务侧损失）
- **未命中样例**：前 10 条没进 top-k 的 query 及其 gold，方便人工核对标注

评测集格式（`data/eval/queries.jsonl`）：

```json
{"query":"是他们发错颜色了，寄回去的快递钱还要我掏吗","gold":["售后政策::退货运费"],"answerable":true}
{"query":"支持以旧换新吗","gold":[],"answerable":false}
```

## 外部数据集验证（题不是自己出的）

上面那套集是自己出题、自己标 gold、还在同一份上选阈值，只能内部对比。所以另外接了一份公开数据：
[`C-MTEB/T2Reranking`](https://huggingface.co/datasets/C-MTEB/T2Reranking) 的 6,129 条中文
query / positive / hard-negative 三元组，抽 200 条查询、组织成 3,000 条语料（正例 + 难负例），转成检索任务。

| 指标 | 哈希向量 | text-embedding-v3 | 差异 |
| --- | ---: | ---: | ---: |
| recall@3 | 0.6450 | **0.8450** | +20.0pp |
| recall@5 | 0.7050 | **0.9000** | +19.5pp |
| MRR | 0.5657 | **0.7205** | +0.155 |

方向与内部集一致（+20pp vs +18.5pp），说明内部结论不是自己出题刷出来的。
局限写在 [docs/external-eval.md](docs/external-eval.md)：小语料抽样让绝对值偏乐观、只测检索不测拒答、
200 条查询下 1 条题就是 0.5pp。

复现：

```bash
pip install "pyarrow<17"   # pyarrow 17+ 要 numpy>=2，与本项目的 numpy<2 冲突
python scripts/prepare_cmteb.py --dataset T2Reranking --queries 200 --corpus 3000
ragagent --config config.offline.toml --db data/external/T2Reranking/hash.db \
         ingest-jsonl data/external/T2Reranking/passages.jsonl
ragagent --config config.offline.toml --db data/external/T2Reranking/hash.db \
         eval --dataset data/external/T2Reranking/queries.jsonl --no-sweep --report reports/ext-hash.md
```

> 注意：32 条内部集能看出**检索层**的配置差异，但样本量不足以外推线上精度；
> 拒答层的数字只能说明「该信号在口语化输入上不成立」，不要引用成拒答能力。
> 换 embedding 模型后必须 `ragagent ingest data/docs --force` 重建索引，否则新旧向量不在同一空间。
> 判断向量到底有没有生效看 `ragagent stats` 的 `vector_active` / `vectors`——**不要只看 `embedder` 名字**，
> 配置写了 `enable_vector = true` 不等于库里真有向量（缺向量会打 WARNING 并降级为 BM25+短语）。
> 想换分词器：`pip install -e ".[zh]"` 后设 `RAGAGENT_TOKENIZER=jieba`（换分词器后同样需重新 ingest）。

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
├── data/eval/            # 内部评测集
├── docs/eval-report-example.md  # 内部评测报告示例
├── docs/external-eval.md        # 外部数据集验证（C-MTEB）
├── scripts/prepare_cmteb.py     # 外部评测集预处理
└── tests/                # pytest
```

## 设计取舍（面试可直接讲）

1. **不用 LangChain**：切分、融合、精排、评测都需要可解释的中间分数，框架会把这些藏起来。
2. **不用向量数据库**：万级 chunk 内存算余弦足够；接口留了 `VectorIndex`，换 pgvector 只改一个文件。
3. **RRF 而不是加权**：三路分数量纲不同，RRF 只看排名，鲁棒；加权只在明确调过参时用。
4. **检索和拒答分开度量**：`recall@k` 只算检索，`recall@k_effective` 再叠加拒答。早期把两者混在一个数字里，
   结果「检索变差」和「拒答更激进」分不清——同一个 recall 下降，可能是召回坏了，也可能只是阈值调紧了。
5. **拒答是合法动作，但当前信号有明确缺陷**：一是「未见实词比例」做主要信号、阈值由评测取舍决定；
   二是已在口语化评测集上验证出它**不成立**（可答 0.66 vs 不可答 0.78，分布重叠，无可用阈值）。
   诚实的说法是「这条路径需要换掉」，而不是「已验证可靠」。
6. **父子块**：子块保证召回精度，父块保证回答上下文完整。
7. **Agent 只给工具出口**：检索/读全文/对比/提交答案都是工具，工具之外不能产生副作用。
8. **网页只用一个静态文件**：演示要看的是「引用、分数、trace」，不是 SPA 工程；不引入 Node 构建链，
   页面和 API 同源，`serve` 一条命令就能跑，也不影响 CLI 和接口。

## 测试

```bash
pytest -q
```

覆盖：切分（FAQ/表格/长文）、BM25 排序、RRF 融合、评测指标、Agent 多步循环（FakeChat 注入）、
静态页挂载（能打开且不挡 API 路由）、向量状态可见性（缺向量时告警而不是静默降级）、
外部语料批量入库（一条记录 = 一个 chunk）、**gold 只认章节名**的口径回归。

## 实现规模

- 34 个 Python 文件、约 3,530 行（其中测试 8 个文件约 400 行 / 30 个用例，脚本 1 个约 160 行），
  外加一个约 1,120 行的静态页
- 纯 Python 后端 + 一个 HTML 文件；无 Node 构建链、无外部数据库服务：SQLite 单文件即可跑

## 已知限制

- `config.offline.toml` 的哈希向量没有语义能力，只用于零依赖跑通；`config.toml` 已接真实 embedding（text-embedding-v3）；
- **拒答信号在口语化输入上失效**（误拒率 0.48、拒答准确率 0.6，且换更好的向量无改善）——已知缺陷，需换信号，不是调阈值能修；
- 内部评测集 32 条：足以区分**检索层**的配置差异（recall@3 0.778 vs 0.963），但不足以外推线上精度；
- 外部验证用的是 3,000 条小语料抽样（`C-MTEB/T2Reranking`），recall 绝对值比百万级语料乐观，且只覆盖检索层；
- SQLite 单机，多租户/高并发需换 PostgreSQL + pgvector；
- 未做鉴权、限流、重试与结构化日志，属演示级服务，别当生产网关用。
