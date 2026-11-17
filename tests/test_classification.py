"""Tests for the classification service."""

from ml_pipeline.services.classification import classify_species
from ml_pipeline.services.detection import detect_animals


def test_classify_species() -> None:
    detection = detect_animals(b"fake jpeg")[0]
    classifications = classify_species(b"fake jpeg", detection)

    assert [(c.genus, c.species) for c in classifications] == [
        ("Rattus", "rattus"),
        ("Rattus", "norvegicus"),
    ]
    # most likely candidate first
    assert classifications[0].confidence > classifications[1].confidence
    assert classifications[0].family == "Muridae"
    assert classifications[0].common_name == "black rat"
