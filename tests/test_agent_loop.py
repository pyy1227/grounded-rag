"""Agent 循环测试：工具调用、提交答案、步数耗尽兜底。"""
from __future__ import annotations

import json

from ragagent.agent.loop import AgentRunner
from ragagent.llm import LLMResult, ToolCall
from ragagent.retrieval.retriever import Retriever


class ScriptedChat:
    name = "scripted"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=None, max_tokens=None):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def _search_call(query: str = "退货运费谁承担", call_id: str = "c1") -> LLMResult:
    return LLMResult(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="search_knowledge",
                arguments=json.dumps({"query": query}, ensure_ascii=False),
            )
        ]
    )


def _submit_call(answer: str, citations=None, refused=False, call_id="c2") -> LLMResult:
    return LLMResult(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="submit_answer",
                arguments=json.dumps(
                    {"answer": answer, "citations": citations or [], "refused": refused},
                    ensure_ascii=False,
                ),
            )
        ]
    )


def test_agent_searches_then_submits(temp_env):
    config, storage = temp_env
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call(), _submit_call("质量问题由商家承担。")])
    answer = AgentRunner(storage, config, chat=chat, retriever=retriever).run("退货运费谁承担")

    assert answer.refused is False
    assert "商家承担" in answer.text
    assert answer.citations, "没有显式引用时应回填检索结果"
    assert any(event.get("tool") == "search_knowledge" for event in answer.trace)


def test_agent_refusal_is_honored(temp_env):
    config, storage = temp_env
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call("量子计算机维修"), _submit_call("无法回答", refused=True)])
    answer = AgentRunner(storage, config, chat=chat, retriever=retriever).run("量子计算机维修")
    assert answer.refused is True
    assert answer.citations == []


def test_agent_falls_back_when_steps_exhausted(temp_env):
    config, storage = temp_env
    config.agent.max_steps = 2
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call(), _search_call(query="退货运费")])
    answer = AgentRunner(storage, config, chat=chat, retriever=retriever).run("退货运费谁承担")
    assert answer.mode == "agent-fallback"
    assert answer.text
    assert answer.citations


def test_agent_without_any_retrieval_refuses(temp_env):
    config, storage = temp_env
    retriever = Retriever(storage, config)

    class DirectAnswer:
        name = "direct"

        def chat(self, *args, **kwargs):
            return LLMResult(content="我直接回答，没有检索。", tool_calls=[])

    answer = AgentRunner(storage, config, chat=DirectAnswer(), retriever=retriever).run("退货运费谁承担")
    assert answer.refused is True
    assert answer.citations == []


class StubGrounding:
    name = "stub"

    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.calls = 0

    def check(self, answer, contexts):
        self.calls += 1
        assert contexts, "接地校验必须拿到引用/检索上下文"
        return self.supported, "stub-reason"


def test_agent_grounding_blocks_fabricated_answer(temp_env):
    """Agent 和 /chat 一样要过接地校验：编造的答案应降级为摘录。"""
    config, storage = temp_env
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call(), _submit_call("退货运费一律由买家承担，并额外赔付 500 元。")])
    grounding = StubGrounding(supported=False)

    answer = AgentRunner(
        storage, config, chat=chat, retriever=retriever, grounding=grounding
    ).run("退货运费谁承担")

    assert grounding.calls == 1
    assert answer.grounded is False
    assert answer.mode == "agent-fallback"
    assert "商家承担" in answer.text, "降级后应摘录原文"
    assert any(event.get("action") == "grounding_failed" for event in answer.trace)


def test_agent_grounding_keeps_supported_answer(temp_env):
    config, storage = temp_env
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call(), _submit_call("质量问题由商家承担。[1]")])
    grounding = StubGrounding(supported=True)

    answer = AgentRunner(
        storage, config, chat=chat, retriever=retriever, grounding=grounding
    ).run("退货运费谁承担")

    assert answer.grounded is True
    assert answer.mode == "agent"
    assert "商家承担" in answer.text


def test_agent_require_citation_can_be_relaxed(temp_env):
    """关闭 require_citation 后允许无引用的自由文本，但仍要求先检索。"""
    config, storage = temp_env
    config.agent.require_citation = False
    retriever = Retriever(storage, config)
    chat = ScriptedChat([_search_call(), LLMResult(content="质量问题由商家承担。")])

    answer = AgentRunner(storage, config, chat=chat, retriever=retriever).run("退货运费谁承担")

    assert answer.refused is False
    assert answer.citations == []
