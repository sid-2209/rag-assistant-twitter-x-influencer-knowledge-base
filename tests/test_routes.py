from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_query_endpoint():
    resp = client.post("/query", json={"query": "Who talks about AI?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)