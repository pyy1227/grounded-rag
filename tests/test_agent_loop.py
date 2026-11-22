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
