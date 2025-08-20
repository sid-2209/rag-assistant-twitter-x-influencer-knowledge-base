from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from app.config import VECTOR_PERSIST_DIR


client = TestClient(app)


def test_ingest_endpoint_ingests_sample_dataset() -> None:
    response = client.post("/ingest", json={"dataset_path": "data/raw/sample.json"})
    assert response.status_code == 200
    assert response.json() == {"status": "ingested", "count": 3}


def test_query_endpoint_returns_match_for_tech_keyword() -> None:
    # Ensure data is ingested first
    ingest_resp = client.post("/ingest", json={"dataset_path": "data/raw/sample.json"})
    assert ingest_resp.status_code == 200

    response = client.post("/query", json={"query": "tech"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body
    # Expect at least one match for 'tech'
    assert body["answer"] != "No influencers found"


def test_healthz_endpoint() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vector_store_persistence_roundtrip(tmp_path) -> None:
    # Use the running app to ingest, which should save to disk
    resp = client.post("/ingest", json={"dataset_path": "data/raw/sample.json"})
    assert resp.status_code == 200

    # Verify persisted files exist
    manifest = (VECTOR_PERSIST_DIR / "manifest.json").exists()
    metadata = (VECTOR_PERSIST_DIR / "metadata.json").exists()
    assert manifest and metadata


def test_ingest_with_missing_file_returns_500() -> None:
    response = client.post("/ingest", json={"dataset_path": "data/raw/DOES_NOT_EXIST.json"})
    # Current behavior: raises and returns 500; adjust if app changes to 400 later
    assert response.status_code == 500


def test_feedback_endpoint_records_feedback() -> None:
    response = client.post(
        "/feedback", json={"query_id": "123", "rating": "up"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "feedback recorded"}
