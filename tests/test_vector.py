from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def ingest_sample() -> None:
    resp = client.post("/ingest", json={"dataset_path": "data/raw/sample.json"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ingested"
    assert body["count"] >= 1


def test_vector_search_ai_startups_returns_expected() -> None:
    ingest_sample()
    r = client.post("/query", json={"query": "AI startups"})
    assert r.status_code == 200
    data = r.json()
    names = [c["name"] for c in data.get("citations", [])]
    assert any("Aarav" in n for n in names), f"Expected Aarav, got {names}"


def test_vector_search_workout_returns_expected() -> None:
    ingest_sample()
    r = client.post("/query", json={"query": "workout routine"})
    assert r.status_code == 200
    data = r.json()
    names = [c["name"] for c in data.get("citations", [])]
    assert any("Sanya" in n for n in names), f"Expected Sanya, got {names}"


def test_vector_search_crypto_returns_expected() -> None:
    ingest_sample()
    r = client.post("/query", json={"query": "crypto market"})
    assert r.status_code == 200
    data = r.json()
    names = [c["name"] for c in data.get("citations", [])]
    assert any("Kabir" in n for n in names), f"Expected Kabir, got {names}"
