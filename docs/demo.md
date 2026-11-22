# 演示指南

两种模式，按场景选：

| 模式 | 命令 | 需要 Key | 耗时 | 用在哪 |
| --- | --- | --- | --- | --- |
| 离线 | `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Offline -Pause -FullEval` | 不需要、不联网 | ~10 秒 | 给评审/面试官看"零依赖能跑" |
| 在线 | `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Pause` | `.env` 里的 DeepSeek + DashScope | ~30 秒 | 展示真实模型效果 |

> Windows 默认执行策略禁止运行未签名脚本，所以命令前面带 `-ExecutionPolicy Bypass`。在 PowerShell 里也可以先 `Set-ExecutionPolicy -Scope Process Bypass` 再直接 `.\scripts\demo.ps1`。

脚本会按七幕顺序执行，每幕先打印命令再打印输出，加 `-Pause` 时每幕结束等你按回车。

## 七幕与预期输出

### 第 1 幕 索引现状 `stats`

```json
{ "documents": 5, "chunks": 40, "embedder": "remote", "db": "data/rag.db" }
```

讲点：5 篇演示文档切成 40 个子块，进 SQLite 单文件；`embedder` 字段一眼看出用的哪套向量。

### 第 2 幕 入库幂等 `ingest data/docs`

第一次 `"added": [5 个文件]`，第二次 `"skipped": [5 个文件]`、`"chunks": 0`。

讲点：靠 sha256 判断文件有没有变，没变就跳过——既避免重复 embedding 计费，也保证重复执行安全。

### 第 3 幕 检索可解释 `search "退货运费谁承担" -k 3`

每条命中同时带四层分数：

```json
{
  "score": 0.7785,            // 精排后的最终排序分
  "bm25": 44.1862,            // 一路：词法召回
  "vector": 0.8333,           // 二路：余弦相似度
  "fused": 1.0,               // RRF 融合分
  "reasons": ["bm25#1", "vector#1", "原文短语命中"]
}
```

讲点：这就是"不用 LangChain"的理由——框架会把中间分数藏起来，这里全留着，每条命中都能解释为什么排第一。

### 第 4 幕 带引用回答 `ask "退货运费谁承担"`

```json
{
  "answer": "退货运费由谁承担取决于退货原因……[1][2]",
  "citations": [
    { "index": 1, "doc_title": "售后政策", "section": "常见问题" },
    { "index": 2, "doc_title": "售后政策", "section": "退货运费" }
  ],
  "mode": "llm", "latency_ms": 1710
}
```

讲点：`mode=llm` 说明真模型生效（离线模式会是 `extractive`）；`[n]` 只允许指向真实 chunk，越界编号直接丢弃，没有有效引用就按拒答处理。

### 第 5 幕 拒答 `ask "支持以旧换新吗"`

```json
{ "refused": true, "citations": [], "answer": "抱歉，知识库里没有找到足够的信息……" }
```

讲点：拒答是合法动作，但**别把它讲成已经解决了**。主信号是"问题里的实词在知识库里出现过的比例"，阈值靠扫描选点：`max_unseen_ratio` 从 1.0 扫到 0.5，拒答准确率 0.0 → 1.0，代价是有效 recall@5 从 0.85 掉到 0.22、误拒率 0.11 涨到 0.78——**没有一个点能同时兼顾**。被问到就直说：这套信号在口语化输入上不成立（可答 0.66 vs 不可答 0.78，分布重叠），下一步要换信号。主动交代比被问出来强。

### 第 6 幕 Agent 多步 `agent "退货时效和换货时效有什么区别"`

trace 里能看到：`search_knowledge` → `open_chunk`（读父块全文）→ `submit_answer`，最终 `mode=agent`，citations 校验通过。

讲点：模型唯一的动作出口是工具，工具之外不能产生副作用；步数用尽或全程没检索就自动兜底为拒答/摘录。

### 第 7 幕 评测 `eval`

```
检索层  recall@3 0.7778 / recall@5 0.8148 / MRR 0.6142   （离线哈希向量）
检索层  recall@3 0.9630 / recall@5 0.9630 / MRR 0.8457   （真实向量）
拒答层  误拒率 0.4815 / 拒答准确率 0.6                    （两套完全一样）
外部集  recall@3 0.6450 → 0.8450 / MRR 0.5657 → 0.7205    （C-MTEB，题不是自己出的）
```

加 `-FullEval` 会额外跑 `max_unseen_ratio` 阈值扫描，报告写到 `reports/`。

讲点：评测把检索和拒答拆开看，这是这套 harness 最关键的一步。检索层真实向量领先 15~23 个百分点——
因为题面是口语改写（"是他们发错颜色了，寄回去的快递钱还要我掏吗"），字面匹配吃不住，语义向量能命中。
拒答层两套一模一样，因为信号来自 BM25 词表、跟向量无关；而且两边都差，这是已知缺陷。
最后补一句外部验证：换成 C-MTEB 的公开中文数据（200 查询 / 3,000 语料），差距是 +20 个百分点，
说明这套结论不是自己出题自己打分刷出来的。**我知道哪一层好、哪一层坏、以及为什么**——
这比一个笼统的"准确率 90%"值钱。

---

## 3 分钟精简版

```powershell
ragagent ask "退货运费谁承担"        # 能干对事，带引用
ragagent ask "支持以旧换新吗"        # 没依据敢拒答
ragagent eval --report reports/eval.md
```

## HTTP 接口演示

```powershell
ragagent serve --port 8000
```

打开 http://127.0.0.1:8000/docs ，展开 `/agent` → Try it out → 请求体 `{"query":"发票什么时候能开"}`。
`/health` 会显示 `llm` 和 `embedder`，可以直接证明模型真的接上了。

## 翻车点

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 脚本无法运行、提示未签名 | Windows 执行策略 | 命令前加 `-ExecutionPolicy Bypass` |
| `ask` 很慢或超时 | 模型侧网络 | 提前预热一次；`config.toml` 里 LLM `timeout=60` |
| 输出里中文乱码 | 终端编码 | 用 Windows Terminal；脚本已自动 `chcp 65001` |
| 现场 `ingest --force` 很慢 | 重新 embedding | 演示用不带 `--force`，正好展示 `skipped` 幂等 |
| Agent 不调工具、直接回答 | 用了 `deepseek-reasoner` | 换成 `deepseek-chat`（支持 function calling） |

演示时不要用管道重定向输出（`| Tee-Object` 之类），否则日志会和"第 N 幕"提示交错，屏幕上看不清。
