"""LLM 客户端。

- OpenAIChat：调用 OpenAI 兼容的 /chat/completions，支持 function calling；
- DemoChat：无 API 时的规则型替身，用来演示/测试 Agent 链路。

两者实现同一个 Protocol，Agent 层不需要知道用的是哪个。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .config import AppConfig


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_message(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMResult:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult: ...


class OpenAIChat:
    """OpenAI 兼容客户端（httpx，同步）。"""

    def __init__(self, config: AppConfig) -> None:
        self.url = config.llm.url
        self.model = config.llm.model
        self.api_key = config.llm_api_key()
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.timeout = config.llm.timeout
        self.name = f"openai:{self.model}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        import httpx

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}

        calls: List[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{len(calls)}"),
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                )
            )
        return LLMResult(content=message.get("content") or "", tool_calls=calls, raw=data)


class DemoChat:
    """无 API Key 时的规则型替身。

    行为：第一次调用检索，第二次把 Top-1 结果整理成 submit_answer。
    它不理解语义，只用来证明「工具链路 + 引用」是通的。
    """

    name = "demo:rule-based"

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        self.calls += 1
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = str(message.get("content") or "")
                break

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        tool_names = {(tool.get("function") or {}).get("name") for tool in (tools or [])}

        if not tool_messages:
            if "search_knowledge" in tool_names:
                return LLMResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"demo_search_{self.calls}",
                            name="search_knowledge",
                            arguments=json.dumps({"query": user_message}, ensure_ascii=False),
                        )
                    ],
                )
            return LLMResult(content="暂时无法回答这个问题。", tool_calls=[])

        payload = _load_json(tool_messages[-1].get("content"))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            return LLMResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"demo_refuse_{self.calls}",
                        name="submit_answer",
                        arguments=json.dumps(
                            {"answer": "知识库里没有找到相关信息。", "citations": [], "refused": True},
                            ensure_ascii=False,
                        ),
                    )
                ],
            )

        top = results[0]
        snippet = str(top.get("snippet") or "").strip()
        snippet = _strip_section_prefix(snippet)
        answer = snippet[:160]
        return LLMResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"demo_submit_{self.calls}",
                    name="submit_answer",
                    arguments=json.dumps(
                        {
                            "answer": answer,
                            "citations": [top.get("chunk_id")],
                            "refused": False,
                        },
                        ensure_ascii=False,
                    ),
                )
            ],
        )


def _load_json(text: Any) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(str(text or ""))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_section_prefix(text: str) -> str:
    if text.startswith("（章节："):
        index = text.find("）")
        if index >= 0:
            return text[index + 1 :].strip()
    return text


def build_chat(config: AppConfig) -> ChatModel:
    """有 key 走真实模型，没有 key 用规则替身（并在调用侧标注 mode）。"""
    if config.llm_api_key():
        return OpenAIChat(config)
    return DemoChat()


__all__ = ["ToolCall", "LLMResult", "ChatModel", "OpenAIChat", "DemoChat", "build_chat"]
