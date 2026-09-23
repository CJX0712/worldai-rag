# -*- coding: utf-8 -*-
"""API layer tests via FastAPI TestClient (offline providers)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from worldai.api import create_app  # noqa: E402
from worldai.assembly import build_system  # noqa: E402
from worldai.config import Config  # noqa: E402


def _client():
    cfg = Config(
        embed_provider="hash",
        store_provider="memory",
        rerank_provider="rrf",
        llm_provider="mock",
    )
    system = build_system(cfg)
    return TestClient(create_app(system))


def test_health_reports_providers():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["providers"] == {
        "embed": "hash",
        "store": "memory",
        "rerank": "rrf",
        "llm": "mock",
    }


def test_ingest_and_chat_flow():
    c = _client()
    r = c.post(
        "/ingest",
        json={"texts": ["地球是太阳系第三颗行星，围绕太阳公转。"]},
    )
    assert r.status_code == 200
    assert r.json()["ingested_chunks"] >= 1
    r = c.post("/chat", json={"question": "地球围绕什么转"})
    assert r.status_code == 200
    body = r.json()
    assert "太阳" in body["answer"]
    assert body["provider"] == "mock"
    assert body["citations"]


def test_ingest_validation():
    c = _client()
    assert c.post("/ingest", json={}).status_code == 400
    assert c.post("/chat", json={"question": "  "}).status_code == 400


def test_agent_endpoint_calc():
    c = _client()
    r = c.post("/agent", json={"question": "12*12"})
    assert r.status_code == 200
    assert "144" in r.json()["answer"]


def test_eval_endpoint_deterministic():
    c = _client()
    c.post("/ingest", json={"texts": ["干扰文档：量子引力与弦论。"]})
    m1 = c.get("/eval").json()
    m2 = c.get("/eval").json()
    assert m1 == m2
    assert m1["recall@4"] >= 0.8
