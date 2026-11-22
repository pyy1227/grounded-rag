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
