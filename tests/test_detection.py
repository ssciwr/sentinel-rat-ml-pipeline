"""Tests for detection and classification services."""

from ml_pipeline.schemas.analysis import AnalyzeResponse
from ml_pipeline.services.analysis import analyze_images
from ml_pipeline.services.classification import classify_species
from ml_pipeline.services.detection import detect_animals


def test_detect_animals() -> None:
    detections = detect_animals("/fake/a.jpg")
    assert [d.detected_class for d in detections] == ["animal", "person"]
    assert detections[0].confidence == 0.92
    # classification is a separate step
    assert all(d.classifications == [] for d in detections)


def test_classify_species() -> None:
    detection = detect_animals("/fake/a.jpg")[0]
    classifications = classify_species("/fake/a.jpg", detection)

    assert [(c.genus, c.species) for c in classifications] == [
        ("Rattus", "rattus"),
        ("Rattus", "norvegicus"),
    ]
    # most likely candidate first
    assert classifications[0].confidence > classifications[1].confidence
    assert classifications[0].family == "Muridae"
    assert classifications[0].common_name == "black rat"


def test_analyze_images() -> None:
    response = analyze_images(["/fake/a.jpg", "/fake/b.jpg"])
    assert isinstance(response, AnalyzeResponse)
    assert [r.image_path for r in response.results] == ["/fake/a.jpg", "/fake/b.jpg"]

    result = response.results[0]
    assert result.detection_model.name == "megadetector"
    assert result.classification_model.name == "rodent-classifier"
    animal, person = result.detections
    assert [c.species for c in animal.classifications] == ["rattus", "norvegicus"]
    # only animals are classified
    assert person.classifications == []
