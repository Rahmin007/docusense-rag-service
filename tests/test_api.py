from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["chunks"] > 0


def test_query_contract():
    response = client.post(
        "/api/query",
        json={"question": "What is the database backup retention period?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"answer", "sources", "tokens_used"}
    assert payload["sources"]
    assert {"chunk_id", "similarity_score", "text_snippet", "source_file"} <= set(payload["sources"][0])
