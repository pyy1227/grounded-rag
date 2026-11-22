# RAG 评测报告（示例）

> 由 `ragagent eval --report reports/eval.md` 生成后整理。
> 数据为仓库自带评测集，**只用于配置对比，不代表线上效果**。

- 语料：15 篇文档 / 121 个 chunk（售后、物流、发票、会员、支付、优惠券、账户、维修、促销、3 个商品参数）
- 评测集：44 条，按 6:4 分层拆成 dev 26（23 可答 + 3 不可答）与 test 18（16 可答 + 2 不可答）
- 题面为**口语化改写**，不照抄知识库原词（如「是他们发错颜色了，寄回去的快递钱还要我掏吗」）
- 运行模式：BM25 + text-embedding-v3 + 短语召回 → RRF → heuristic 精排 → 语义拒答校验
- 分词：bigram

## 检索层

同一套检索参数，只换 embedding（test 未参与任何调参）：

| 指标 | dev · 哈希向量 | dev · 真实向量 | test · 真实向量 |
| --- | ---: | ---: | ---: |
| recall@3 | 0.6087 | **0.9565** | 0.8750 |
| recall@3（文档级） | 0.8261 | **1.0000** | 0.8750 |
| recall@5 | — | **1.0000** | 0.8750 |
| recall@5（文档级） | — | **1.0000** | 0.9375 |
| MRR | 0.5862 | **0.8152** | 0.8125 |

哈希向量本质是字符 bigram 的袋，口语改写一上就掉；语义向量能靠 paraphrase 命中，dev 上差 30+ 个百分点。

两个口径并存的原因：`recall@k` 要求命中 gold 指定的章节，`recall@k_doc` 只要求命中 gold 所在文档。
同一答案在 15 篇语料里可能散落在两处（比如"货到付款"在《发货与物流》和《支付与结算》都写了），
文档级口径用来抵消这类标注歧义；本集里两者的差额已通过补全 gold 收窄。

## 拒答层：为什么换掉了「数未见词」

先看四种**词面**信号在 dev 上的上限（balanced accuracy =（1 − 误拒 + 拒答准确）/ 2）：

| 拒答信号 | 最佳 balanced | 该点误拒率 | 该点拒答准确率 |
| --- | ---: | ---: | ---: |
| 未见实词比例（旧：全量 token 等权） | 0.67 | 0.67 | 1.00 |
| 未见实词比例（新：滤虚词 + bigram 0.5 权） | 0.70 | 0.59 | 1.00 |
| top-1 精排分数 | 0.57 | 0.85 | 1.00 |
| top-1 覆盖度 | 0.72 | 0.56 | 1.00 |
| 覆盖度 ∧ 未见比例（联合） | 0.63 | 0.15 | 0.40 |

结论：词面信号在这份口语化集上**分不开**可答与不可答——它们度量的是措辞，不是"资料里有没有答案"。

降噪本身有效，但只解决了一半问题：

| 指标 | 旧信号 | 新信号（滤虚词 + bigram 降权） |
| --- | ---: | ---: |
| 可答问题的比例（p50） | 0.67 | **0.21** |
| 可答问题的比例（max） | 1.00 | **0.58** |
| 不可答问题的比例（p50） | 0.75 | 0.24 |
| 两类是否可分 | 重叠 | **仍重叠** |

噪声降下去了（可答样本不再被虚词顶到 0.8 以上），但不可答样本一起降下来，间距没变。

所以最终改成**语义校验**：把问题与前 3 个检索段落交给模型，问「仅凭这些资料能否回答」。

| 配置（dev） | 误拒率 | 拒答准确率 |
| --- | ---: | ---: |
| 纯词面信号（`verify_mode = "off"`，`max_unseen_ratio = 0.8`） | 0.4815 | 0.60 |
| **语义校验（`verify_mode = "llm"`）** | **0.0435** | **1.00** |

留出集（test，18 条）：误拒率 0.1875、拒答准确率 1.00 —— 未参与调参，略低于 dev 属正常。

## 局限

- dev 只有 23 条可答样本，1 条题 = 4.3 个百分点；test 只有 16 条，1 条 = 6.3 个百分点。别抠小数点后两位；
- 语义校验多一次模型调用（约 +1s），评测里按 `(问题, 前 3 个 chunk)` 缓存；模型判定本身有噪声；
- 离线模式没有 Key，只能退回词面信号，此时拒答准确率为 0（实测误拒 0、拒答准确 0，等于从不拒答）；
- 语料仍是演示级（15 篇 / 121 chunk），检索指标比真实业务库乐观。

## 复现

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY
ragagent ingest data/docs --force

# 调参用 dev；test 是留出集，只在最终报告里跑一次
ragagent eval --dataset data/eval/queries-dev.jsonl --report reports/eval-dev.md
ragagent eval --dataset data/eval/queries-test.jsonl --no-sweep --report reports/eval-test.md
```

没有 Key 时用离线配置（哈希向量 + 词面拒答）：

```bash
ragagent --config config.offline.toml ingest data/docs --force
ragagent --config config.offline.toml eval --dataset data/eval/queries-dev.jsonl --report reports/eval-dev-offline.md
```
