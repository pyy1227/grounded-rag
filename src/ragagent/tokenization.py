"""统一分词。

两种模式：
- bigram（默认）：中文按「单字 + 相邻二字」切分，无依赖、结果确定；
- jieba：词典分词，长文本和真实语料通常更好，需要 `pip install -e ".[zh]"`。

用环境变量切换：`RAGAGENT_TOKENIZER=jieba`（或 auto = 有 jieba 就用）。

为什么默认 bigram：仓库自带的 FAQ 语料上，bigram 的 recall 反而更高
（见 docs/eval-report-example.md 的对比），而且零依赖、可复现。
评测时应该把分词器当作一个可消融变量，而不是写死。
"""
from __future__ import annotations

import os
import re
from typing import List

_CJK = r"\u4e00-\u9fff"
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RUN_RE = re.compile(f"[{_CJK}]+")
_SPACE_RE = re.compile(r"[\s\u3000]+")

try:  # pragma: no cover - 取决于环境
    import jieba

    jieba.setLogLevel(20)
    _JIEBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _JIEBA_AVAILABLE = False

_REQUESTED = os.environ.get("RAGAGENT_TOKENIZER", "bigram").strip().lower()
_USE_JIEBA = _JIEBA_AVAILABLE and _REQUESTED in {"auto", "jieba"}


def tokenize(text: str) -> List[str]:
    """返回用于 BM25/哈希向量的 token 列表。"""
    normalized = normalize(text)
    if not normalized:
        return []

    if _USE_JIEBA:
        return [token for token in jieba.cut_for_search(normalized) if token.strip()]

    tokens: List[str] = []
    tokens.extend(_ASCII_WORD_RE.findall(normalized))
    for run in _CJK_RUN_RE.findall(normalized):
        tokens.extend(list(run))
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


# 虚词与语气助词：这些词在不在知识库里，跟"能不能回答"无关。
# 拒答判定只看实词，否则口语化的问法会被大量虚词稀释。
_STOP_CHARS = frozenset(
    "的了着过吗呢吧啊呀嘛哦哇么什怎们我你他她它这那哪谁"
    "是有在能会要就都还也才又很太把被给让从到对和与或但而及等之其所以为于个些点"
)


def content_tokens(text: str) -> List[str]:
    """拒答判定专用的 token：去掉虚词后，只取单字与**相邻实词**组成的二字词。

    和 `tokenize` 的区别有两处，都是为了降低噪声：
    1. 先滤掉语气助词/虚词（吗、了、能、我……），它们与可答性无关；
    2. bigram 只在原本相邻的实词之间生成——`tokenize` 那种直接按字符滑窗会在
       「拆都没拆」里切出 `拆没`、`没拆` 这种跨虚词组合，几乎必然未命中。
    """
    normalized = normalize(text)
    if not normalized:
        return []

    tokens: List[str] = []
    for run in _CJK_RUN_RE.findall(normalized):
        previous = ""
        for char in run:
            if char in _STOP_CHARS:
                previous = ""
                continue
            tokens.append(char)
            if previous:
                tokens.append(previous + char)
            previous = char
    tokens.extend(_ASCII_WORD_RE.findall(normalized))
    return tokens


def mode() -> str:
    return "jieba" if _USE_JIEBA else "bigram"


__all__ = ["tokenize", "normalize", "mode", "content_tokens"]
