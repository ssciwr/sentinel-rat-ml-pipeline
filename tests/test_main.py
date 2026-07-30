"""Tests for the ML Pipeline app."""

from fastapi.testclient import TestClient

from ml_pipeline.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={"image_path": "/fake/path.jpg"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["animals_detected"] == 2
    assert data["species"] == {"cat": 1, "rodent": 1}
    assert data["confidence"] == 0.95
