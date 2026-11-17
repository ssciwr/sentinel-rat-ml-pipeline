"""Tests for the detection service."""

from ml_pipeline.services.detection import detect_animals


def test_detect_animals() -> None:
    detections = detect_animals(b"fake jpeg")
    assert [d.detected_class for d in detections] == ["animal", "person"]
    assert detections[0].confidence == 0.92
    # classification is a separate step
    assert all(d.classifications == [] for d in detections)
