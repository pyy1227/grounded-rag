"""容错 JSON 解析：模型输出经常带代码块或前后解释。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def load_lenient(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    candidates = [raw]
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    balanced = _find_object(raw)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _find_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


__all__ = ["load_lenient", "truncate"]
