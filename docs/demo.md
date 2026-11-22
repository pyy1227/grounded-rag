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

讲点：拒答是合法动作，**而且现在能说清它是怎么修的**。原来数"问题里的实词在库里出现过没有"：词面信号在口语化输入上两类样本分布重叠（可答 0.66 vs 不可答 0.78），换阈值只能二选一——误拒一半，或一条都不拒。我先做降噪（滤掉语气助词/虚词、bigram 只在相邻实词间生成并降权 0.5），可答样本的比例中位数从 0.67 降到 0.21，误拒从 0.48 降到接近 0，**但区分度没变好**；所以最终换成模型语义校验（仅凭检索到的段落能否回答）：误拒 0.48 → 0.04、拒答准确率 0.6 → 1.0。离线没有 Key 时会退回词面信号，那时拒答能力等于没有——这点要主动说。

### 第 6 幕 Agent 多步 `agent "退货时效和换货时效有什么区别"`

trace 里能看到：`search_knowledge` → `open_chunk`（读父块全文）→ `submit_answer`，最终 `mode=agent`，citations 校验通过。

讲点：模型唯一的动作出口是工具，工具之外不能产生副作用；步数用尽或全程没检索就自动兜底为拒答/摘录。

### 第 7 幕 评测 `eval`

```
语料    15 篇文档 / 121 chunk；评测集 57 条（dev 34：23 可答 + 11 不可答；test 23：16 + 7）
检索层  dev  recall@3 0.6957 → 0.9130 / 文档级 0.8696 → 0.9130 / MRR 0.5667 → 0.7464
        test recall@3 0.9375 / 文档级 1.0000 / MRR 0.9115
拒答层  旧词面：误拒 0.26 / 拒答准确 0.27；语义校验：误拒 0.1304 / 拒答准确 1.00（11/11）
外部集  recall@3 0.6450 → 0.8450 / MRR 0.5657 → 0.7205（C-MTEB，题不是自己出的）
```

加 `-FullEval` 会额外跑 `max_unseen_ratio` 阈值扫描，报告写到 `reports/`。

讲点：这套 harness 有三个关键决定。第一，检索和拒答分开度量，否则"召回坏了"和"阈值调紧了"混成一个数字。第二，有留出集——阈值只在 dev 上选，test 只报一次。第三，不可答样本不能太少：一开始只有 5 条，1 条题就是 20～33 个百分点，那个"拒答准确率"根本站不住；补到 18 条（属性/政策/凭证/服务四类）之后才敢报。
检索层真实向量领先 20+ 个百分点，因为题面是口语改写（"是他们发错颜色了，寄回去的快递钱还要我掏吗"），字面匹配吃不住、语义向量能命中。拒答层原来两套向量一样差，说明短板不在向量而在信号；换成模型语义校验后误拒降到 13%、拒答准确率 1.00。离线没有 Key 时会退回词面信号，那时拒答能力等于没有——这点要主动说。
最后补外部验证：C-MTEB 公开中文数据（200 查询 / 3,000 语料）上差距 +20 个百分点，说明结论不是自己出题自己打分。
**我知道哪一层好、哪一层坏、为什么、以及怎么修的**——这比一个笼统的"准确率 90%"值钱。

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
