"""Tests for detection and classification services."""

from ml_pipeline.schemas.analysis import AnalyzeResponse
from ml_pipeline.services.detection import detect_animals


def test_detect_animals() -> None:
    response = detect_animals("/fake/path.jpg")
    assert isinstance(response, AnalyzeResponse)
    assert response.animals_detected == 2
    assert response.species == {"cat": 1, "rodent": 1}
    assert response.confidence == 0.95
