"""Tests for the ML Pipeline app."""

import pytest
from fastapi.testclient import TestClient

from ml_pipeline.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.usefixtures("watch_folder")
def test_analyze_endpoint() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={"image_paths": ["hd-zoo-01/a.jpg", "hd-zoo-01/b.jpg"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["image_path"] for r in results] == ["hd-zoo-01/a.jpg", "hd-zoo-01/b.jpg"]

    result = results[0]
    assert result["detection_model"]["name"] == "megadetector"
    assert result["detection_model"]["version"] == "v5a"
    assert result["classification_model"]["name"] == "rodent-classifier"
    assert result["classification_model"]["version"] == "0.1"
    detection = result["detections"][0]
    assert detection["detected_class"] == "animal"
    assert detection["confidence"] == 0.92
    assert detection["bbox"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5}
    assert [c["species"] for c in detection["classifications"]] == [
        "rattus",
        "norvegicus",
    ]
    assert detection["classifications"][0]["family"] == "Muridae"
    assert result["detections"][1]["detected_class"] == "person"
    assert result["detections"][1]["classifications"] == []


@pytest.mark.usefixtures("watch_folder")
def test_analyze_endpoint_leaves_out_missing_images() -> None:
    response = client.post(
        "/api/v1/analyze",
        json={"image_paths": ["hd-zoo-01/missing.jpg", "hd-zoo-01/a.jpg"]},
    )
    assert response.status_code == 200
    assert [r["image_path"] for r in response.json()["results"]] == ["hd-zoo-01/a.jpg"]


def test_analyze_endpoint_rejects_empty_list() -> None:
    response = client.post("/api/v1/analyze", json={"image_paths": []})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_single_path() -> None:
    response = client.post("/api/v1/analyze", json={"image_path": "/fake/a.jpg"})
    assert response.status_code == 422
