"""从 C-MTEB 生成一份小规模外部检索评测集。

自带那 32 条题是自己出的、gold 也是自己标的，只能做内部对比；外部数据集的题
不是我们写的，用来验证结论不是自己给自己打分。

用的数据是 C-MTEB/T2Reranking（中文，6,129 条 query / positive / hard negative 三元组）。
它本来是 reranking 任务，这里把正例和难负例组织成一份小语料，转成检索任务：
语料 = 抽样查询的正例 + 其他查询的正例或难负例随机补齐（默认 3,000 条）；
评测 = 每条查询的 gold 就是它自己的正例 passage。
这样既有真标注，也不依赖 qrels（C-MTEB 的 retrieval 子集只放了 corpus 和 queries）。

用法：
    python scripts/prepare_cmteb.py --dataset T2Reranking --queries 200 --corpus 3000

依赖 pyarrow<17（17+ 要求 numpy>=2，本项目锁了 numpy<2）。
parquet 从 hf-mirror 下载并缓存到 data/external/cache/，已存在就复用。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    raise SystemExit('需要 pyarrow：pip install "pyarrow<17"')

CACHE = Path("data/external/cache")
MIRROR = "https://hf-mirror.com"
TAG_RE = re.compile(r"<[^>]{1,40}>")


def _download(dataset: str, cache: Path) -> Path:
    """从镜像拉取 parquet 到本地缓存；已存在则直接用。返回落地的文件路径。"""
    cache.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache.glob("*.parquet"))
    if existing:
        print(f"复用缓存 {existing[0].name}")
        return existing[0]

    listing = httpx.get(
        f"{MIRROR}/api/datasets/C-MTEB/{dataset}", timeout=60, follow_redirects=True
    ).json()
    files = [
        item["rfilename"]
        for item in listing.get("siblings") or []
        if item["rfilename"].endswith(".parquet")
    ]
    if not files:
        raise SystemExit(f"C-MTEB/{dataset} 下没有 parquet 文件")

    target = files[0]
    dest = cache / Path(target).name
    print(f"下载 {target}")
    url = f"{MIRROR}/datasets/C-MTEB/{dataset}/resolve/main/{target}"
    with httpx.stream("GET", url, timeout=900, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 16):
                fh.write(chunk)
    size_mb = dest.stat().st_size / 1048576
    print(f"  完成 {size_mb:.1f} MB" + (f" / {total / 1048576:.1f} MB" if total else ""))
    return dest


def _clean(text: Any) -> str:
    """去掉 <br>/<img> 之类的 HTML 片段并压缩空白。"""
    value = TAG_RE.sub(" ", str(text or ""))
    return " ".join(value.split())


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _first_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, (str, int))]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 C-MTEB 小规模外部检索评测集")
    parser.add_argument("--dataset", default="T2Reranking")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--corpus", type=int, default=3000)
    parser.add_argument("--max-chars", type=int, default=1500,
                        help="单条 passage 截断长度（原数据最长 6.8 万字，超 embedding 输入上限）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache = CACHE / args.dataset
    parquet = _download(args.dataset, cache)
    rows = _read_rows(parquet)
    print(f"原始样本 {len(rows)} 条，列：{sorted(rows[0])}")

    # 收集所有 passage：正例和难负例，按清洗后的文本去重
    text_to_pid: Dict[str, str] = {}
    pid_text: List[str] = []

    def register(text: str) -> str:
        if text not in text_to_pid:
            pid = f"p{len(pid_text)}"
            text_to_pid[text] = pid
            pid_text.append(text)
        return text_to_pid[text]

    triples: List[Tuple[str, str]] = []  # (query, positive_pid)
    for row in rows:
        query = _clean(row.get("query"))
        positives = [_clean(item) for item in _first_list(row.get("positive"))]
        negatives = [_clean(item) for item in _first_list(row.get("negative"))]
        positives = [p for p in positives if p]
        if not query or not positives:
            continue
        for passage in positives + negatives:
            if passage:
                register(passage)
        triples.append((query, register(positives[0])))

    print(f"passage 去重后 {len(pid_text)} 条，可用查询 {len(triples)} 条")

    # 抽样：先抽查询，语料由正例和随机补齐的其他 passage 组成
    rng = random.Random(args.seed)
    rng.shuffle(triples)
    sampled = triples[: args.queries]

    keep = {pid for _, pid in sampled}
    corpus_pids = list(keep)
    pool = [pid for pid in (f"p{i}" for i in range(len(pid_text))) if pid not in keep]
    rng.shuffle(pool)
    corpus_pids.extend(pool[: max(0, args.corpus - len(corpus_pids))])
    corpus_set = set(corpus_pids)
    print(f"评测语料 {len(corpus_pids)} 条（正例 {len(keep)} 条）")

    # 写出
    out = Path("data/external") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    truncated = 0
    with (out / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for pid in corpus_pids:
            text = pid_text[int(pid[1:])]
            if len(text) > args.max_chars:
                truncated += 1
                text = text[: args.max_chars]
            fh.write(json.dumps(
                {"doc_id": pid, "title": "passage", "text": text},
                ensure_ascii=False,
            ) + "\n")

    written = 0
    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for query, pid in sampled:
            if pid not in corpus_set:
                continue
            fh.write(json.dumps(
                {"query": query, "gold": [f"{pid}::"], "answerable": True},
                ensure_ascii=False,
            ) + "\n")
            written += 1

    manifest = {
        "dataset": f"C-MTEB/{args.dataset}",
        "source": "https://huggingface.co/datasets/C-MTEB/" + args.dataset,
        "mirror": "https://hf-mirror.com",
        "task": "query→passage 检索（由 reranking 三元组组织而成）",
        "seed": args.seed,
        "queries": written,
        "corpus": len(corpus_pids),
        "max_chars": args.max_chars,
        "truncated_passages": truncated,
        "note": (
            "小语料抽样（正例 + 难负例），recall 绝对值比全量语料乐观；"
            "用于横向比较配置，不代表线上指标。"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 {out}/passages.jsonl（{len(corpus_pids)} 条）、queries.jsonl（{written} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
