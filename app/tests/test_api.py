from fastapi.testclient import TestClient

from app.main import app as fa_app

client = TestClient(fa_app)


def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"response": "pong"}
