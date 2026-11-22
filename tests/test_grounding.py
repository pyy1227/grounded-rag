"""接地校验测试：答案没被引用资料支持时，降级为摘录而不是硬答。"""
from __future__ import annotations

from ragagent.config import AppConfig
from ragagent.generation import AnswerGenerator
from ragagent.indexer import Indexer
from ragagent.llm import LLMResult
from ragagent.retrieval.retriever import Retriever
from ragagent.storage import Storage
from ragagent.verify import build_grounding

DOC = """# 售后政策

## 退货运费

| 退货原因 | 运费承担方 |
| --- | --- |
| 商品质量问题 | 商家承担 |
| 7 天无理由 | 买家承担 |

## 常见问题

Q：退货运费谁承担？
A：质量问题由商家承担，7 天无理由由买家承担。
"""


class FakeChat:
    """按 prompt 区分「回答问题」和「接地校验」两种调用。"""

    name = "fake"

    def __init__(self, answer: str, supported: bool) -> None:
        self.answer = answer
        self.supported = supported
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=None, max_tokens=None):
        self.calls += 1
        prompt = " ".join(str(message.get("content") or "") for message in messages)
        if "事实主张" in prompt:
            payload = '{"supported": %s, "unsupported": [], "reason": "test"}' % str(
                self.supported
            ).lower()
        else:
            payload = self.answer
        return LLMResult(content=payload)


def _generator(workdir, chat) -> AnswerGenerator:
    docs = workdir / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "售后政策.md").write_text(DOC, encoding="utf-8")

    config = AppConfig()
    config.storage.path = str(workdir / "rag.db")
    config.retrieval.enable_vector = False
    storage = Storage(config.storage.path)
    Indexer(storage, config).ingest_path(docs)

    retriever = Retriever(storage, config)
    retriever.verifier = None  # 只测接地校验，不掺拒答
    generator = AnswerGenerator(storage, config, retriever=retriever, chat=chat)
    generator.grounding = build_grounding(chat)
    return generator


def test_grounded_answer_is_kept(workdir):
    chat = FakeChat(answer="质量问题由商家承担 [1]。", supported=True)
    answer = _generator(workdir, chat).answer("退货运费谁承担")
    assert answer.mode == "llm"
    assert answer.grounded is True
    assert answer.refused is False


def test_unsupported_answer_falls_back_to_extract(workdir):
    """模型编了资料里没有的内容时，应降级为摘录并标记 grounded=False。"""
    chat = FakeChat(answer="退货运费一律由买家承担，并且额外赔付 50 元 [1]。", supported=False)
    answer = _generator(workdir, chat).answer("退货运费谁承担")
    assert answer.mode == "extractive"
    assert answer.grounded is False
    assert answer.refused is False
    assert "承担" in answer.text  # 摘录来自原文
