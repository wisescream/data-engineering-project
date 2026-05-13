from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_chat_payload_validation():
    # Empty payload should fail
    response = client.post("/chat", json={})
    assert response.status_code == 422
