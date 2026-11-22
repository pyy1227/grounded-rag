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

## 前置条件

| 项 | 要求 | 说明 |
| --- | --- | --- |
| Python | **3.11 或更高** | `pyproject.toml` 要求 `>=3.11`；CI 覆盖 3.11 / 3.12 |
| 依赖管理 | `uv`（推荐） | `pyproject.toml` 声明依赖，`uv.lock` 锁版本；没装 uv 也可以 `pip install -e ".[dev]"`，只是不锁版本 |
| 模型 Key | **可以没有** | 在线配置需要 `DEEPSEEK_API_KEY` + `DASHSCOPE_API_KEY`；没有 Key 就用 `config.offline.toml`，全流程离线可跑 |
| 网络 | 可选 | 在线配置需能访问 `api.deepseek.com`、`dashscope.aliyuncs.com`；离线配置不联网 |
| 操作系统 | Windows / Linux / macOS | 演示脚本是 PowerShell（Windows 自带 5.1+），其余功能跨平台 |
| 中文分词 | 可选 | 默认字符 bigram；装 jieba 并设 `RAGAGENT_TOKENIZER=jieba` 才切换成词典分词 |

装依赖：

```bash
uv sync --extra dev

# 没装 uv 的机器
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # 装项目和开发依赖，版本交给 pyproject 解析
```

`pip install -e ".[dev]"` 不会按 `uv.lock` 锁版本；要复现同一套版本，用 uv。

装完执行 `ragagent --help`，应能列出 `ingest / search / ask / agent / eval / stats / ingest-jsonl / serve` 8 个子命令。

可选的中文分词（jieba 在 `[zh]` extra 里，`uv sync --extra zh` 或 `pip install jieba`）：

```bash
export RAGAGENT_TOKENIZER=jieba    # Windows: $env:RAGAGENT_TOKENIZER="jieba"
```

## 快速开始

```bash
cd rag-agent
uv sync --extra dev          # 没装 uv 见上「前置条件」

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

## 文档入库

检索和问答都建立在索引之上，所以第一步必须把文档写进数据库。索引是 SQLite 单文件，
位置由 `storage.path` 指定（默认 `data/rag.db`）。

### 支持的格式

| 扩展名 | 解析方式 |
| --- | --- |
| `.md` / `.markdown` / `.txt` | 按 Markdown 标题切章节，识别表格行 |
| `.pdf` | 按页抽取文本 |
| `.docx` | 按标题样式切章节，表格转 Markdown |
| `.xlsx` | 每个 sheet 转一张 Markdown 表格 |

其他扩展名不会被静默忽略：会作为失败项写进 `errors`。

### 命令

```bash
# 目录（递归子目录）；不写路径时默认就是 data/docs
ragagent ingest data/docs

# 单个文件也接受
ragagent ingest data/docs/售后政策.md

# 强制重建，见下文「什么时候必须 --force」
ragagent ingest data/docs --force

# 没有 Key 时用离线配置（哈希向量，不联网）
ragagent --config config.offline.toml ingest data/docs

# 写一份独立索引，不动 data/rag.db
ragagent --db data/other.db ingest ./my_docs
```

实测输出（15 篇 demo 文档、离线配置）：

```json
{
  "added": ["data\\docs\\优惠券与促销.md", "…"],
  "updated": [],
  "skipped": [],
  "errors": [],
  "chunks": 121,
  "elapsed_ms": 1298,
  "stats": {
    "documents": 15,
    "attempted": 15,
    "errors": 0,
    "children": 121,
    "kinds": {"total": 121, "faq": 62, "table": 19, "avg_chars": 1503},
    "embedder": "hash-256-bigram",
    "tokenizer": "bigram"
  }
}
```

`documents` 只数**成功**的篇数，解析失败的文件只出现在 `errors` 里；只要有失败，CLI 退出码就是 1（HTTP 侧返回 422）。

### 入库做了什么

1. **解析**：按扩展名取出「章节 + 段落 + 表格」结构，不是整篇塞进一个块；
2. **切分**：标题 / FAQ 问答 / 表格行切成父子块——子块（`chunking.max_chars`，默认 520）进索引保证召回精度，父块（`parent_max_chars`，默认 1600）留给回答补上下文；
3. **幂等**：按文件内容 sha256 判断。内容没变 → `skipped`，不会重复调用 embedding 计费；内容变了 → `updated`，先删该文档的旧 chunk 再写新的；
4. **向量化**：按 `embedding.batch_size`（默认 10，DashScope text-embedding-v3 的单次上限）分批请求。这一步失败不会写半截数据：该文件整体进 `errors`，索引保持入库前的状态；
5. **写库**：chunk、向量、文档元数据写进 SQLite，`chunks` 就等于 `stats.children`。

### 验证入库结果

```bash
ragagent stats                          # documents / chunks / vector_active / vectors_loaded
ragagent search "退货运费谁承担" -k 5    # 能召回就说明索引可用
```

看两个数就够：`chunks > 0` 表示文档进库了；`vector_active = true` 表示向量也生效了。
只看 `embedder` 名字不算数——配置写了 `enable_vector = true`、库里却没有向量时，
检索会静默降级成 BM25 + 短语（`stats` 里 `vectors_loaded` 会是 0）。

### 什么时候必须 `--force`

下面这些改动会让库里的旧向量和新配置不在同一个空间，不加 `--force` 会命中旧 chunk：
看着「有结果」，语义其实是错的。

- 换 embedding 模型（`embedding.provider` / `embedding.model`）；
- 换分词器（`RAGAGENT_TOKENIZER` 在 jieba / bigram 之间切换）；
- 调切分参数（`chunking.*`）。

等价做法是删掉 `data/rag.db` 再完整 ingest 一次。

### 批量导入外部语料（一条记录 = 一个 chunk）

公开评测集（如 C-MTEB）本身就是 query-passage 对，不需要再解析切分，走旁路命令：

```bash
ragagent --config config.offline.toml --db data/external/T2Reranking/hash.db ingest-jsonl data/external/T2Reranking/passages.jsonl
```

每行一个 JSON：`{"doc_id": "...", "title": "...", "text": "..."}`，其中 `doc_id` 和 `text` 必填。

### 通过 HTTP 入库

`ragagent serve` 起来后也能触发：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"path": "data/docs"}'
```

路径必须落在 `RAGAGENT_INGEST_ROOT`（默认 `data/docs`）内，越界返回 400，路径不存在返回 404。

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

## HTTP 安全边界

`serve` 默认只绑 `127.0.0.1`。服务端不会因为一个路径参数就读写整个磁盘：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `RAGAGENT_INGEST_ROOT` | `data/docs` | `/ingest` 只接受该目录内的路径（CLI 不受限） |
| `RAGAGENT_EVAL_ROOT` | `data/eval` | `/eval` 的 dataset 必须在该目录内 |
| `RAGAGENT_REPORT_ROOT` | `reports` | `/eval` 的 report 必须写在该目录内 |
| `RAGAGENT_API_TOKEN` | 空（不鉴权） | 设置后 `/ingest`、`/eval` 需要 `X-API-Token` 或 `Authorization: Bearer` |

越界路径返回 400，入库有失败文件返回 422，token 不匹配返回 401。仍然没有用户体系、
限流和审计日志——这是演示级服务，别直接挂到公网。

## 配置要点

| 配置 | 说明 |
| --- | --- |
| `retrieval.fusion` | `rrf`（默认，只看排名）或 `weighted`（可调权重） |
| `retrieval.rerank_mode` | `heuristic` / `llm` / `noop`，用于消融对比 |
| `retrieval.refuse_threshold` | 精排分低于该值直接拒答（次要信号） |
| `retrieval.max_unseen_ratio` | 问题里「知识库没出现过的实词」占比阈值，越小越容易拒答（仅离线词面模式生效） |
| `retrieval.verify_mode` / `verify_snippets` / `verify_retry_top_k` | 语义拒答校验：判定窗口和 `need_more` 重试窗口；`off` 时退回词面信号 |
| `retrieval.grounding_mode` | 生成后的接地校验；`off` 时关闭 |
| `chunking.max_chars` / `overlap_chars` | 切分粒度，影响召回和上下文完整度 |
| `embedding.provider` | `hash`（离线）或 `remote`（OpenAI 兼容 /embeddings） |
| `embedding.batch_size` | 单次 embed 请求的条数，默认 10（DashScope text-embedding-v3 上限，超了 400） |
| `api_key_env` | key 从哪个环境变量读。`config.toml` 指向 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` |
| `llm.*` | 回答/Agent 用的模型；**不要用 `deepseek-reasoner`**，它不支持 function calling |

## 评测（实测数字）

```bash
ragagent eval --dataset data/eval/queries-dev.jsonl                       # 调参用 dev
ragagent eval --dataset data/eval/queries-test.jsonl --no-sweep           # 留出集，只跑一次
```

语料 15 篇文档 / 121 个 chunk（商品参数、售后、物流、发票、会员、支付、优惠券、账户、维修、促销）。
评测集 57 条、按 6:4 分层拆成 **dev 34 条**（23 可答 + 11 不可答）和 **test 23 条**（16 可答 + 7 不可答）：
阈值只在 dev 上选，test 是留出集，避免"在同一份集上调参再报分"。
题面是**口语化改写**，不照抄知识库原词（例如「是他们发错颜色了，寄回去的快递钱还要我掏吗」）。

评测把**检索**和**拒答**分开度量：`recall@k` 只算检索命中，`recall@k_effective` 再叠加拒答过滤，
`recall@k_doc` 只要求命中 gold 所在**文档**（同一答案散落在多个章节时更公平）。

**检索层**（同一份集、同一套检索参数，只换 embedding）：

| 指标 | dev · 哈希向量 | dev · text-embedding-v3 | test · text-embedding-v3 |
| --- | ---: | ---: | ---: |
| recall@3 | 0.6957 | **0.9130** | 0.9375 |
| recall@3（文档级） | 0.8696 | **0.9130** | 1.0000 |
| recall@5 | 0.8261 | **0.9130** | 1.0000 |
| MRR | 0.5667 | **0.7464** | 0.9115 |

哈希向量本质是字符 bigram 的袋，口语改写一上就掉；语义向量能靠 paraphrase 命中，dev 上差 20+ 个百分点。

gold 只认 `section_path`，**不认正文**：早期版本写成「章节名 or 正文含关键字」，会让「见退货运费表格」
这类只是引用别处的段落也算命中，机制上注定虚高。

**拒答层：从「数未见词」换成「模型语义校验」**

早期用「问题里未见实词的比例」做拒答，但在口语化评测集上两类样本分布重叠，
任何阈值都要误拒一半以上。词面信号都试过（含联合判定），dev 上的上限：

| 拒答信号 | 最佳 balanced | 该点的误拒率 | 该点的拒答准确率 |
| --- | ---: | ---: | ---: |
| 未见实词比例（旧：全量 token 等权） | 0.70 | 0.61 | 1.00 |
| 未见实词比例（新：滤虚词 + bigram 降权） | 0.74 | 0.52 | 1.00 |
| **语义校验（现，`verify_mode = "llm"`）** | 0.96 | **0.0870** | **1.00** |

语义校验没有可扫的阈值，只有一个工作点；它的 balanced 按（1 − 误拒率 + 拒答准确率）/ 2 算出。

同一份 dev、同一个默认阈值（`max_unseen_ratio = 0.8`）下：

| 配置 | 误拒率 | 拒答准确率 |
| --- | ---: | ---: |
| 纯词面（旧信号） | 0.26 | 0.27 |
| 纯词面（降噪后新信号） | 0.00 | 0.00 |
| **语义校验（现）** | **0.0870** | **1.00** |

语义校验分三档，不是为了更严，而是为了**区分「资料里没有」和「这次没检索到」**：

- `answerable`：资料能回答核心诉求（答得不全也算这一档）→ 直接作答；
- `need_more`：缺回答所必需的关键事实 → **扩大窗口二次检索**（`verify_retry_top_k`，默认 10）再判一次；
- `cannot`：资料无关或确实没有 → 拒答。

本集上重试没有救回案例（`need_more` 的样本重试后依然 `need_more`），所以三档的实际收益是
"把不确定显式标出来、并留出重试通道"，不是立刻降误拒；重试路径有单测覆盖。

**接地校验（生成后）**：答案生成后再让模型检查"每条事实主张能否在引用段落里找到依据"，
找不到就把答案降级为摘录并标 `grounded=false`。这是防幻觉的一层，和拒答是两件事：

| 场景（关掉拒答、强行生成） | 接地校验的表现 |
| --- | --- |
| 18 条不可答问题 | 12 条模型自己说"资料不足"（无引用→按拒答处理）、**1 条被接地拦下**、5 条漏过 |
| 39 条可答问题 | 38 条正常、1 条被误伤（降级为摘录） |

漏过的 5 条都是"答了相关问题但不是所问"——接地校验按定义查的是**主张是否有依据**，
查不出"答非所问"。后者是 answerability 校验的职责（实测 11/11 全对）。

代价是多一次模型调用（评测里按 `(问题, 前 5 个 chunk)` 缓存，同一问题只算一次），
延迟增加约 1s；离线模式没有这个调用，也**没有可用的拒答能力**（误拒 0、拒答准确 0，等于不拒）。

不可答样本从 5 条补到 18 条（属性/政策/凭证/服务四类），语义校验在 dev 上 11/11，
test 上 6/7（漏掉「会员可以转给别人吗」：检索到会员等级/优惠券章节，模型判成可答）。
补样本之前"拒答准确率"只有 3～5 条分母，1 条题就是 20～33 个百分点，数字本身不成立。

**换一批数据还准吗？** 做了一次跨库实验：把 15 篇文档切成「产品参数」和「政策制度」两半，
在只有产品参数的库里问政策问题（10 条）、在只有政策的库里问产品问题（4 条）——答案物理上不存在：

| 场景 | 拒答 | 说明 |
| --- | ---: | --- |
| 政策库 × 产品问题 | 4/4 | 全部正确拒答 |
| 产品库 × 政策问题 | 10/10 | 全部正确拒答 |
| 本库 × 本库问题 | 误拒 1/14 | 那 1 条不是判错：答案排在第 4 位，校验器只看了前 3 条 |

也就是说，**拒答能力并不依赖校准用题的措辞**，但和检索强耦合：答案没进"校验器看得见的窗口"，
就会被当成"资料里没有"。把窗口从 3 条放宽到 5 条后，dev 误拒 3/23 → 2/23（0.087）、test 从 6.25% → 0%，
负样本仍然 11/11 全拒。这条耦合是下一步要根治的地方（见已知限制）。

完整报告见 [docs/eval-report-example.md](docs/eval-report-example.md)。

报告包含：

- `recall@3` / `recall@5`：gold 是否进入前 k（**不经过拒答**，纯检索）
- `recall@k_doc`：只要求命中 gold 所在文档，抵消章节级标注歧义
- `recall@k_effective`：叠加拒答后的有效召回，两者之差即拒答的代价
- `MRR`：首个命中排名的倒数均值
- `refusal_accuracy` / `over_refusal`：拒答准确率与误拒率
- **阈值扫描**：`max_unseen_ratio` 的取舍表（只在离线词面模式生成；语义校验模式会自动跳过，因为该阈值不参与判定）
- **未命中样例**：前 10 条没进 top-k 的 query 及其 gold，方便人工核对标注

> `MRR` 按请求的 top-k 截断后计算。`need_more` 重试会临时扩大检索窗口，早期版本把扩窗后的
> 第 6–10 名也算进 MRR，数字会略高；现在评测层先截断再算，检索指标不再受拒答策略影响。

评测集格式（`data/eval/queries-dev.jsonl`）：

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

方向与内部集一致（外部 +20pp vs 内部 dev +21.7pp），说明内部结论不是自己出题刷出来的。
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

> 注意：内部 dev 只有 34 条（可答 23），能看出**检索层**的配置差异，但样本量不足以外推线上精度。
> 拒答能力依赖模型校验（有 Key 才有）；离线模式下的词面拒答已实测不可用，别当成能力。
> 换 embedding 模型后必须 `ragagent ingest data/docs --force` 重建索引，否则新旧向量不在同一空间。
> 判断向量到底有没有生效看 `ragagent stats` 的 `vector_active` / `vectors` / `vectors_loaded` / `vector_retry_pending`——**不要只看 `embedder` 名字**，
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
│   ├── verify.py         # 语义拒答校验（能否仅凭检索段落回答）
│   ├── evaluation.py     # 指标 + 阈值扫描 + 报告
│   ├── agent/            # 工具调用 Agent（多步 + trace）
│   ├── api.py            # FastAPI
│   ├── cli.py            # 命令行
│   └── static/index.html # 网页（单文件原生 JS，无构建）
├── data/docs/            # demo 文档（15 篇）
├── data/eval/            # 内部评测集（queries-dev / queries-test）
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
5. **拒答先降噪、再上语义校验**：原先用「未见实词比例」，在口语化输入上两类样本分布重叠（可答 0.66 vs 不可答 0.78），
   换阈值救不了。先做了降噪（滤掉语气助词/虚词、bigram 只在相邻实词间生成并降权到 0.5）——
   可答样本的比例中位数从 0.67 降到 0.21，误拒随之从 0.48 降到几乎为 0，但**区分度没变好**（两类仍重叠）；
   所以最终把拒答交给模型语义校验（仅凭检索段落能否回答），词面信号只做离线兜底。误拒 0.48 → 0.087（默认阈值）、拒答准确 0.6 → 1.0。
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
外部语料批量入库（一条记录 = 一个 chunk）、**gold 只认章节名**的口径回归、
**拒答信号**（虚词过滤、相邻 bigram、语义校验优先于词面信号）、
**回答失败兜底**（模型报错也要摘录原文，而不是只回一句错误）、
**Agent 接地校验**（编造答案降级为摘录）、**入库失败退出码**、
**路径白名单 / API token**、**扩窗结果按 k 截断后再算 MRR**。

## 实现规模

- 37 个 Python 文件、约 4,300 行（src 26 个文件 3,442 行；测试 10 个文件 723 行 / 49 个用例；
  脚本 1 个 163 行），外加一个约 1,130 行的静态页；语料 15 篇文档 / 121 个 chunk
- 纯 Python 后端 + 一个 HTML 文件；无 Node 构建链、无外部数据库服务：SQLite 单文件即可跑

## 已知限制

- `config.offline.toml` 的哈希向量没有语义能力，只用于零依赖跑通；`config.toml` 已接真实 embedding（text-embedding-v3）；
- **离线模式没有可用的拒答能力**：词面信号已实测分不开可答/不可答（balanced 上限 0.70），只会两种极端——全拒或全不拒；
- 同一份 `config.toml` 在无 Key 时不会立刻报错：查询 embedding 返回 401 后会降级为 BM25+短语
  （实测 dev recall@3 0.7391 / test 0.75），拒答退回词面信号（实测全不拒）。这条降级路径能跑，但不是推荐配置；
- 拒答准确率依赖模型语义校验，多一次 LLM 调用（约 +1s）；模型判定偶有噪声，评测里靠缓存避免重复计费；
- 内部评测集 57 条（dev 34 / test 23；不可答 18 条），足以区分**检索层**的配置差异（dev recall@3 0.70 vs 0.91），但不足以外推线上精度；
- 外部验证用的是 3,000 条小语料抽样（`C-MTEB/T2Reranking`），recall 绝对值比百万级语料乐观，且只覆盖检索层；
- SQLite 单机，多租户/高并发需换 PostgreSQL + pgvector；
- 鉴权只有可选的 `RAGAGENT_API_TOKEN`，没有用户体系；限流、重试、结构化日志仍未做，属演示级服务，别当生产网关用。
