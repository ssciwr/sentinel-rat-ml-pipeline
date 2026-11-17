"""Tests for the analysis of a batch of images."""

import logging

from ml_pipeline.schemas.analysis import AnalyzeResponse
from ml_pipeline.services.analysis import analyze_images


def test_analyze_images(watch_folder) -> None:
    response = analyze_images(["hd-zoo-01/a.jpg", "hd-zoo-01/b.jpg"])
    assert isinstance(response, AnalyzeResponse)
    # the requested, relative paths are returned
    assert [r.image_path for r in response.results] == [
        "hd-zoo-01/a.jpg",
        "hd-zoo-01/b.jpg",
    ]

    result = response.results[0]
    assert result.detection_model.name == "megadetector"
    assert result.classification_model.name == "rodent-classifier"
    animal, person = result.detections
    assert [c.species for c in animal.classifications] == ["rattus", "norvegicus"]
    # only animals are classified
    assert person.classifications == []


def test_analyze_images_skips_unreadable(watch_folder, caplog) -> None:
    with caplog.at_level(logging.ERROR):
        response = analyze_images(
            ["hd-zoo-01/missing.jpg", "hd-zoo-01/b.jpg", "../outside.jpg"]
        )
    # the readable image is still analyzed
    assert [r.image_path for r in response.results] == ["hd-zoo-01/b.jpg"]
    assert "Skipping hd-zoo-01/missing.jpg" in caplog.text
    assert "Skipping ../outside.jpg" in caplog.text


def test_analyze_images_none_readable(watch_folder) -> None:
    assert analyze_images(["hd-zoo-01/missing.jpg"]).results == []
