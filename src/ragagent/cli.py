"""命令行入口：ingest / search / ask / agent / eval / serve / stats。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

from .config import AppConfig

# 读取项目根目录的 .env（已存在的环境变量优先，不会被覆盖）
load_dotenv()


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build(config: AppConfig):
    from .storage import Storage

    storage = Storage(config.storage.path)
    return storage


def _load_config(args: argparse.Namespace) -> AppConfig:
    """加载配置，支持 `--db` 覆盖库路径（同一份配置对比不同索引时用）。"""
    config = AppConfig.load(args.config)
    db = getattr(args, "db", None)
    if db:
        config.storage.path = db
    return config


def cmd_ingest(args: argparse.Namespace) -> int:
    from .indexer import Indexer

    config = _load_config(args)
    storage = _build(config)
    indexer = Indexer(storage, config)
    try:
        report = indexer.ingest_path(args.path, force=args.force)
    except (OSError, ValueError) as exc:
        print(f"入库失败: {exc}", file=sys.stderr)
        return 1
    _print(report)
    if report.get("errors"):
        for item in report["errors"]:
            print(f"入库失败: {item['path']}: {item['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .retrieval.retriever import Retriever

    config = _load_config(args)
    storage = _build(config)
    hits = Retriever(storage, config).search(args.query, top_k=args.top_k)
    _print({"query": args.query, "hits": [hit.to_dict() for hit in hits]})
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .generation import AnswerGenerator

    config = _load_config(args)
    storage = _build(config)
    answer = AnswerGenerator(storage, config).answer(args.query, top_k=args.top_k)
    _print(answer.to_dict())
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from .agent.loop import AgentRunner

    config = _load_config(args)
    storage = _build(config)
    answer = AgentRunner(storage, config).run(args.query)
    _print(answer.to_dict())
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import evaluate, load_dataset, sweep_thresholds, write_report
    from .retrieval.retriever import Retriever

    config = _load_config(args)
    if args.threshold is not None:
        config.retrieval.refuse_threshold = args.threshold
    storage = _build(config)
    dataset = args.dataset or config.evaluation.dataset
    if not Path(dataset).exists():
        print(f"评测集不存在: {dataset}", file=sys.stderr)
        return 1

    cases = load_dataset(dataset)
    retriever = Retriever(storage, config)
    result = evaluate(
        retriever,
        cases,
        k_values=config.evaluation.k_values,
        dataset=dataset,
    )
    extra: dict = {}
    if not args.no_sweep:
        if getattr(retriever, "verifier", None) is not None:
            print(
                "语义校验已启用，跳过 max_unseen_ratio 阈值扫描"
                "（该扫描只对离线词面拒答有意义）",
                file=sys.stderr,
            )
        else:
            extra["sweep"] = sweep_thresholds(retriever, cases)
    report_path = write_report(result, args.report, extra)

    storage.save_eval_run(
        run_id=f"{Path(dataset).stem}-{result['metrics']['generated_at']}",
        dataset=dataset,
        config_json=result["config"],
        metrics=result["metrics"],
        report_path=report_path,
    )
    summary = {**result["metrics"], "report": report_path}
    if extra.get("sweep"):
        summary["sweep"] = extra["sweep"]
    _print(summary)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .retrieval.retriever import Retriever
    from .tokenization import mode as tokenizer_mode

    config = _load_config(args)
    storage = _build(config)
    retriever = Retriever(storage, config)
    model = retriever.embedder.model
    status = retriever.index_status()
    vectors_in_db = storage.count_embeddings(model)
    _print(
        {
            "documents": len(storage.list_documents()),
            "chunks": storage.count_chunks(),
            "embedder": config.embedding.provider,
            "embedding_model": model,
            "vector_enabled": config.retrieval.enable_vector,
            # 运行态：embedding 服务失败后这里会变 false
            "vector_active": status["vector_active"],
            "vectors": vectors_in_db,
            "vectors_loaded": status["vectors"],
            "vector_retry_pending": status["retry_pending"],
            "vector_models_in_db": storage.list_embedding_models(),
            "tokenizer": tokenizer_mode(),
            "db": config.storage.path,
            "last_ingest": storage.get_meta("last_ingest_at"),
        }
    )
    return 0


def cmd_ingest_jsonl(args: argparse.Namespace) -> int:
    """外部评测集专用：一条记录当一个 chunk，跳过解析和切分。"""
    from .indexer import Indexer

    config = _load_config(args)
    path = Path(args.path)
    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    storage = _build(config)
    try:
        report = Indexer(storage, config).ingest_records(records)
    except (OSError, ValueError) as exc:
        print(f"记录入库失败: {exc}", file=sys.stderr)
        return 1
    _print(report)
    if report.get("errors"):
        for item in report["errors"]:
            print(f"记录入库失败: {item['path']}: {item['error']}", file=sys.stderr)
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.config:
        import os

        os.environ["RAGAGENT_CONFIG"] = args.config
    uvicorn.run("ragagent.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragagent", description="可评测的 RAG 知识助手")
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    parser.add_argument("--db", default=None, help="覆盖 storage.path（对比不同索引时用）")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入文档")
    ingest.add_argument("path", nargs="?", default="data/docs")
    ingest.add_argument("--force", action="store_true", help="忽略 sha256 缓存强制重建")
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="只检索，不生成")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="RAG 回答")
    ask.add_argument("query")
    ask.add_argument("-k", "--top-k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)

    agent = sub.add_parser("agent", help="Agent 多步回答")
    agent.add_argument("query")
    agent.set_defaults(func=cmd_agent)

    eval_cmd = sub.add_parser("eval", help="跑评测")
    eval_cmd.add_argument("--dataset", default=None)
    eval_cmd.add_argument("--report", default="reports/eval.md")
    eval_cmd.add_argument("--threshold", type=float, default=None)
    eval_cmd.add_argument("--no-sweep", action="store_true")
    eval_cmd.set_defaults(func=cmd_eval)

    stats = sub.add_parser("stats", help="索引统计")
    stats.set_defaults(func=cmd_stats)

    ingest_jsonl = sub.add_parser(
        "ingest-jsonl",
        help="把「一条记录=一个 chunk」的 JSONL 入库（外部评测集用）",
    )
    ingest_jsonl.add_argument("path")
    ingest_jsonl.set_defaults(func=cmd_ingest_jsonl)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
