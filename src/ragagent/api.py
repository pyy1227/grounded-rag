"""FastAPI 服务。

启动：
    uvicorn ragagent.api:app --port 8000
或：
    ragagent serve --port 8000
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent.loop import AgentRunner
from .config import AppConfig
from .evaluation import evaluate, load_dataset, sweep_thresholds, write_report
from .generation import AnswerGenerator
from .indexer import Indexer
from .retrieval.retriever import Retriever
from .storage import Storage

# 必须在读取 RAGAGENT_CONFIG 之前加载，否则 .env 里的配置路径拿不到
load_dotenv()

DEFAULT_CONFIG = os.environ.get("RAGAGENT_CONFIG", "config.toml")
DEFAULT_DOCS = "data/docs"
# 单文件静态页（原生 JS，无构建步骤），挂载在根路径
STATIC_DIR = Path(__file__).resolve().parent / "static"


class IngestRequest(BaseModel):
    path: Optional[str] = Field(None, description="文件或目录，默认 data/docs，且必须位于 RAGAGENT_INGEST_ROOT 内")
    force: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="返回条数（1-20）")

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class SearchRequest(QueryRequest):
    pass


class AskRequest(QueryRequest):
    pass


class EvalRequest(BaseModel):
    dataset: Optional[str] = None
    report: Optional[str] = "reports/eval.md"
    sweep: bool = True


def create_app(config: Optional[AppConfig] = None, config_path: Optional[str] = None) -> FastAPI:
    app_config = config or AppConfig.load(config_path or DEFAULT_CONFIG)
    storage = Storage(app_config.storage.path)
    retriever = Retriever(storage, app_config)
    indexer = Indexer(storage, app_config)
    generator = AnswerGenerator(storage, app_config, retriever)
    agent = AgentRunner(storage, app_config, retriever=retriever)

    # 安全边界：只允许读写配置好的目录，避免 /ingest、/eval 变成任意文件读写
    ingest_root = Path(os.environ.get("RAGAGENT_INGEST_ROOT", DEFAULT_DOCS)).resolve()
    eval_root = Path(os.environ.get("RAGAGENT_EVAL_ROOT", "data/eval")).resolve()
    report_root = Path(os.environ.get("RAGAGENT_REPORT_ROOT", "reports")).resolve()
    api_token = os.environ.get("RAGAGENT_API_TOKEN", "").strip()

    def resolve_within(value: str, root: Path, label: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"{label}必须位于 {root} 内: {value}",
            )
        return resolved

    def require_token(request: Request) -> None:
        """设置了 RAGAGENT_API_TOKEN 才校验，不设置时不鉴权，方便本地开发。"""
        if not api_token:
            return
        provided = request.headers.get("x-api-token", "").strip()
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided or not secrets.compare_digest(
            provided.encode("utf-8"), api_token.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="缺少或错误的 API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(title="RAG Agent", version="0.1.0")
    app.state.config = app_config
    app.state.storage = storage
    app.state.retriever = retriever
    app.state.indexer = indexer
    app.state.generator = generator
    app.state.agent = agent

    @app.get("/health")
    def health() -> Dict[str, Any]:
        from .tokenization import mode as tokenizer_mode

        model = retriever.embedder.model
        status = retriever.index_status()
        vectors_in_db = storage.count_embeddings(model)
        return {
            "status": "ok",
            "documents": len(storage.list_documents()),
            "chunks": storage.count_chunks(),
            "embedder": model,
            "vector_enabled": app_config.retrieval.enable_vector,
            # 运行态：embedding 服务失败后这里会变 false，不能只看配置或库里的向量条数
            "vector_active": status["vector_active"],
            "vectors": vectors_in_db,
            "vectors_loaded": status["vectors"],
            "vector_retry_pending": status["retry_pending"],
            "vector_models_in_db": storage.list_embedding_models(),
            "tokenizer": tokenizer_mode(),
            "llm": getattr(generator.chat, "name", "unknown"),
        }

    @app.post("/ingest", dependencies=[Depends(require_token)])
    def ingest(request: IngestRequest) -> Dict[str, Any]:
        path = resolve_within(request.path or DEFAULT_DOCS, ingest_root, "入库路径")
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在: {request.path or DEFAULT_DOCS}")
        try:
            report = indexer.ingest_path(str(path), force=request.force)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        retriever.invalidate()
        if report.get("errors"):
            raise HTTPException(
                status_code=422,
                detail={"message": "部分或全部文件入库失败", "report": report},
            )
        return report

    @app.get("/documents")
    def documents() -> Dict[str, Any]:
        return {
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "chunks": doc.chunk_count,
                    "version": doc.version,
                    "ingested_at": doc.ingested_at,
                }
                for doc in storage.list_documents()
            ]
        }

    @app.post("/search")
    def search(request: SearchRequest) -> Dict[str, Any]:
        hits = retriever.search(request.query, top_k=request.top_k)
        return {"query": request.query, "hits": [hit.to_dict() for hit in hits]}

    @app.post("/chat")
    def chat(request: AskRequest) -> Dict[str, Any]:
        return generator.answer(request.query, top_k=request.top_k).to_dict()

    @app.post("/agent")
    def agent_endpoint(request: AskRequest) -> Dict[str, Any]:
        return agent.run(request.query).to_dict()

    @app.post("/eval", dependencies=[Depends(require_token)])
    def run_eval(request: EvalRequest) -> Dict[str, Any]:
        dataset = resolve_within(
            request.dataset or app_config.evaluation.dataset, eval_root, "评测集路径"
        )
        if not dataset.exists():
            raise HTTPException(status_code=404, detail=f"评测集不存在: {dataset}")
        # 先校验报告路径，避免跑完评测才发现写不出去
        report_path = (
            str(resolve_within(request.report, report_root, "报告路径"))
            if request.report
            else None
        )
        cases = load_dataset(dataset)
        result = evaluate(retriever, cases, k_values=app_config.evaluation.k_values, dataset=str(dataset))

        # 语义校验优先于词面信号，此时扫 max_unseen_ratio 只会得到一张恒等的表
        sweep_enabled = bool(request.sweep) and getattr(retriever, "verifier", None) is None
        extra = {"sweep": sweep_thresholds(retriever, cases)} if sweep_enabled else {}
        if report_path:
            write_report(result, report_path, extra)
            storage.save_eval_run(
                run_id=f"{Path(dataset).stem}-{result['metrics']['generated_at']}",
                dataset=str(dataset),
                config_json=result["config"],
                metrics=result["metrics"],
                report_path=report_path,
            )
            result["report_path"] = report_path
        result["sweep"] = extra.get("sweep", [])
        result["sweep_skipped"] = bool(request.sweep) and not sweep_enabled
        return result

    # 静态页必须最后挂载：Starlette 按注册顺序匹配，挂到 "/" 会接住所有未命中路径，
    # 放前面会把 /health、/chat 等接口一起吃掉。
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")

    return app


app = create_app()
