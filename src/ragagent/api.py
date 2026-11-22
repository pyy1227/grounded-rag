"""FastAPI 服务。

启动：
    uvicorn ragagent.api:app --port 8000
或：
    ragagent serve --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    path: Optional[str] = Field(None, description="文件或目录，默认 data/docs")
    force: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


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
        vectors = storage.count_embeddings(model)
        return {
            "status": "ok",
            "documents": len(storage.list_documents()),
            "chunks": storage.count_chunks(),
            "embedder": model,
            "vector_enabled": app_config.retrieval.enable_vector,
            # 配置开了向量、库里却没有对应向量时，这里会是 false——不要只看 embedder 名字
            "vector_active": vectors > 0,
            "vectors": vectors,
            "vector_models_in_db": storage.list_embedding_models(),
            "tokenizer": tokenizer_mode(),
            "llm": getattr(generator.chat, "name", "unknown"),
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> Dict[str, Any]:
        path = request.path or DEFAULT_DOCS
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
        report = indexer.ingest_path(path, force=request.force)
        retriever.invalidate()
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

    @app.post("/eval")
    def run_eval(request: EvalRequest) -> Dict[str, Any]:
        dataset = request.dataset or app_config.evaluation.dataset
        if not Path(dataset).exists():
            raise HTTPException(status_code=404, detail=f"评测集不存在: {dataset}")
        cases = load_dataset(dataset)
        result = evaluate(retriever, cases, k_values=app_config.evaluation.k_values, dataset=dataset)
        extra = {"sweep": sweep_thresholds(retriever, cases)} if request.sweep else {}
        if request.report:
            report_path = write_report(result, request.report, extra)
            storage.save_eval_run(
                run_id=f"{Path(dataset).stem}-{result['metrics']['generated_at']}",
                dataset=dataset,
                config_json=result["config"],
                metrics=result["metrics"],
                report_path=report_path,
            )
            result["report_path"] = report_path
        result["sweep"] = extra.get("sweep", [])
        return result

    # 静态页必须最后挂载：Starlette 按注册顺序匹配，挂到 "/" 会兜住所有未命中路径，
    # 放前面会把 /health、/chat 等接口一起吃掉。
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")

    return app


app = create_app()
