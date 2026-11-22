"""HTTP 接口测试（TestClient，不需要起服务）。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from ragagent.api import create_app


def _client(config):
    return TestClient(create_app(config=config))


def test_health_reports_index(temp_env):
    config, _ = temp_env
    response = _client(config).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] >= 1
    assert body["chunks"] >= 1


def test_documents_endpoint(temp_env):
    config, _ = temp_env
    body = _client(config).get("/documents").json()
    assert body["documents"]
    assert body["documents"][0]["doc_id"]


def test_search_chat_and_agent(temp_env):
    config, _ = temp_env
    client = _client(config)

    search = client.post("/search", json={"query": "退货运费谁承担", "top_k": 3}).json()
    assert search["hits"]
    assert "chunk_id" in search["hits"][0]

    chat = client.post("/chat", json={"query": "退货运费谁承担"}).json()
    assert chat["answer"]
    assert chat["citations"]

    agent = client.post("/agent", json={"query": "退货运费谁承担"}).json()
    assert agent["answer"]
    assert agent["trace"]


def test_web_ui_is_served(temp_env):
    """静态页挂在根路径，且不能把 API 路由挡掉。"""
    config, _ = temp_env
    client = _client(config)

    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "企业知识库" in page.text

    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_ingest_rejects_path_outside_allowed_root(temp_env, workdir):
    """/ingest 只能读 RAGAGENT_INGEST_ROOT（默认 data/docs）内的文件。"""
    config, _ = temp_env
    client = _client(config)

    response = client.post("/ingest", json={"path": str(workdir / "outside.md")})
    assert response.status_code == 400
    assert "入库路径" in response.json()["detail"]


def test_eval_report_must_stay_inside_reports_dir(temp_env):
    """/eval 的 report 不能借 ../ 写到仓库外。"""
    config, _ = temp_env
    client = _client(config)

    response = client.post(
        "/eval",
        json={
            "dataset": "data/eval/queries-dev.jsonl",
            "report": "../evil.md",
            "sweep": False,
        },
    )
    assert response.status_code == 400
    assert "报告路径" in response.json()["detail"]


def test_api_token_protects_mutating_endpoints(temp_env, monkeypatch):
    config, _ = temp_env
    monkeypatch.setenv("RAGAGENT_API_TOKEN", "s3cret")
    client = _client(config)
    payload = {"path": "data/docs/售后政策.md"}

    assert client.post("/ingest", json=payload).status_code == 401
    ok = client.post("/ingest", json=payload, headers={"X-API-Token": "s3cret"})
    assert ok.status_code == 200
    assert ok.json()["added"]
