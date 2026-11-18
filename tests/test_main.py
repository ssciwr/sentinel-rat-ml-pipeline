"""Tests for the ML Pipeline HTTP API, run with the tiny service models."""

import pytest
from fastapi.testclient import TestClient

from ml_pipeline.main import app
from tests.service_support import (
    ANIMAL_BOX,
    CLASSIFIER_INFO,
    DETECTOR_INFO,
    HUMAN_BOX,
    RED_TAXON,
    write_two_colour_image,
)


@pytest.fixture
def client() -> TestClient:
    """A test client of the app."""
    return TestClient(app)


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint(client, configured_services, tmp_path) -> None:
    """The response has the README shape: one result per path in order, both models,
    labelled detections, classifications only for animals, and `error` null."""
    paths = [str(write_two_colour_image(tmp_path / name)) for name in ("a.png", "b.png")]

    response = client.post("/api/v1/analyze", json={"image_paths": paths})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["image_path"] for r in results] == paths
    result = results[0]
    assert set(result) == {"image_path", "detection_model", "classification_model", "detections", "error"}
    assert result["detection_model"] == DETECTOR_INFO
    assert result["classification_model"] == CLASSIFIER_INFO
    assert result["error"] is None
    animal, human = result["detections"]
    assert set(animal) == {"detected_class", "confidence", "bbox", "classifications"}
    assert animal["detected_class"] == "animal"
    assert animal["confidence"] == pytest.approx(0.9)
    assert animal["bbox"] == ANIMAL_BOX
    (classification,) = animal["classifications"]
    assert {key: classification[key] for key in RED_TAXON} == RED_TAXON
    assert set(classification) == {
        "kingdom",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species",
        "common_name",
        "confidence",
    }
    assert human["detected_class"] == "human"
    assert human["bbox"] == HUMAN_BOX
    assert human["classifications"] == []


def test_analyze_endpoint_reports_unreadable_image(client, configured_services, tmp_path) -> None:
    """A missing image gets an error and no detections; the request still succeeds."""
    good = str(write_two_colour_image(tmp_path / "good.png"))
    missing = str(tmp_path / "missing.jpg")

    response = client.post("/api/v1/analyze", json={"image_paths": [missing, good]})

    assert response.status_code == 200
    failed, analysed = response.json()["results"]
    assert failed["image_path"] == missing
    assert isinstance(failed["error"], str) and failed["error"]
    assert failed["detections"] == []
    assert analysed["error"] is None
    assert len(analysed["detections"]) == 2


@pytest.mark.parametrize(
    "image_paths",
    [["image.jpg"], ["./images/image.jpg"], ["/data/a.jpg", "relative/b.jpg"], [""], ["/data/a\u0000b.jpg"]],
)
def test_analyze_endpoint_rejects_relative_paths(client, image_paths) -> None:
    """Every path must be absolute (and free of NUL bytes); otherwise the request is invalid (422)."""
    response = client.post("/api/v1/analyze", json={"image_paths": image_paths})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_empty_list(client) -> None:
    response = client.post("/api/v1/analyze", json={"image_paths": []})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_single_path(client) -> None:
    response = client.post("/api/v1/analyze", json={"image_path": "/fake/a.jpg"})
    assert response.status_code == 422
