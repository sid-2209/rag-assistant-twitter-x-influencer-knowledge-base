import pytest
from fastapi.testclient import TestClient
from app.api import app
import os
from app import rag as rag_module

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_sample_dataset():
    # Pre-warm ingestion with the sample dataset
    dataset_path = "data/raw/sample.json"
    assert os.path.exists(dataset_path), "Sample dataset not found"
    resp = client.post("/ingest", json={"dataset_path": dataset_path})
    assert resp.status_code == 200


def test_ai_startup_query():
    resp = client.post("/query", json={"query": "Who are top voices in AI startups?"})
    assert resp.status_code == 200
    data = resp.json()
    # Should generate an answer and cite Aarav
    assert "answer" in data
    assert any("Aarav" in c["name"] for c in data["citations"])


def test_fitness_query():
    resp = client.post("/query", json={"query": "Any advice for fitness routine?"})
    assert resp.status_code == 200
    data = resp.json()
    # Should generate an answer and cite Sanya
    assert "answer" in data
    assert any("Sanya" in c["name"] for c in data["citations"])


def test_unknown_query():
    resp = client.post("/query", json={"query": "What is the best pizza topping?"})
    assert resp.status_code == 200
    data = resp.json()
    # Should gracefully say it doesn’t know (no citations)
    assert "answer" in data
    assert data["citations"] == [] or all(
        c["name"] not in ["Aarav Mehta", "Sanya Kapoor", "Kabir Malhotra"]
        for c in data["citations"]
    )


def test_openai_offline_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(rag_module, "OpenAI", None)

    local_client = TestClient(app)
    ingest_resp = local_client.post("/ingest", json={"dataset_path": "data/raw/sample.json"})
    assert ingest_resp.status_code == 200

    resp = local_client.post("/query", json={"query": "some query"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data.get("citations"), list)