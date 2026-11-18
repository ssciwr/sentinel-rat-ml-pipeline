"""Tests for the detection, classification and analysis services, run with tiny real models.

The `service_session` fixture (tests/conftest.py) runs the tiny-model configs of
tests/service_support.py: a detector with a fixed `animal` box (left half) and `human` box
(right half), and a classifier that classifies by dominant colour.
"""

import pytest
from PIL import Image

from ml_pipeline.schemas.analysis import AnalyzeResponse, Classification, Detection, ModelInfo
from ml_pipeline.services.analysis import analyze_images
from ml_pipeline.services.classification import classify_species
from ml_pipeline.services.detection import detect_animals
from ml_pipeline.services.session import ModelConfig, Session
from tests.service_support import (
    ANIMAL_BOX,
    CLASSIFIER_INFO,
    DETECTOR_INFO,
    HUMAN_BOX,
    RED_TAXON,
    write_broken_detector_config,
    write_non_image,
    write_truncated_jpeg,
    write_two_colour_image,
)


@pytest.fixture
def two_colour_image(tmp_path):
    """Absolute path of a PNG whose left half is red and right half is blue."""
    return write_two_colour_image(tmp_path / "two_colour.png")


# -------------------------------------------------------------- detection / classification


def test_detect_animals_returns_the_detectors_labelled_detections(service_session):
    """The given detector's detections, sorted, labelled and not yet classified."""
    detections = detect_animals(service_session.detector, Image.new("RGB", (16, 8), (0, 255, 0)))

    assert [d.detected_class for d in detections] == ["animal", "human"]
    assert [d.confidence for d in detections] == pytest.approx([0.9, 0.8])
    assert [d.bbox.model_dump() for d in detections] == [ANIMAL_BOX, HUMAN_BOX]
    assert all(isinstance(d, Detection) and d.classifications == [] for d in detections)


def test_classify_species_returns_the_classifiers_classifications(service_session):
    """A red crop is classified as the red taxon, with its configured taxonomy."""
    classifications = classify_species(service_session.classifier, Image.new("RGB", (6, 6), (255, 0, 0)))

    assert len(classifications) == 1
    assert isinstance(classifications[0], Classification)
    assert classifications[0].model_dump(include=set(RED_TAXON)) == RED_TAXON
    assert classifications[0].confidence > 0.99


# ------------------------------------------------------------------------ analyze_images


def test_only_animal_detections_are_classified_from_their_crop(service_session, two_colour_image):
    """The animal box (red half) is cropped and classified as the red taxon only; a crop of
    the wrong region or the whole image would also give blue. The human box keeps []."""
    response = analyze_images([str(two_colour_image)], service_session)

    assert isinstance(response, AnalyzeResponse)
    (result,) = response.results
    animal, human = result.detections
    assert animal.detected_class == "animal"
    assert animal.bbox.model_dump() == ANIMAL_BOX
    assert [c.model_dump(include=set(RED_TAXON)) for c in animal.classifications] == [RED_TAXON]
    assert human.detected_class == "human"
    assert human.classifications == []


def test_results_follow_request_order_with_model_info(service_session, tmp_path):
    """One result per requested path, in request order (duplicates included), each naming
    both models of the session and carrying no error."""
    first = write_two_colour_image(tmp_path / "first.png")
    second = write_two_colour_image(tmp_path / "second.jpg")
    paths = [str(second), str(first), str(second)]

    response = analyze_images(paths, service_session)

    assert [r.image_path for r in response.results] == paths
    for result in response.results:
        assert result.detection_model == ModelInfo(**DETECTOR_INFO)
        assert result.classification_model == ModelInfo(**CLASSIFIER_INFO)
        assert result.error is None
        assert [d.detected_class for d in result.detections] == ["animal", "human"]


@pytest.mark.parametrize(
    "make_bad_path",
    [
        lambda tmp_path: tmp_path / "missing.jpg",
        lambda tmp_path: write_non_image(tmp_path / "notes.jpg"),
        lambda tmp_path: tmp_path,
    ],
    ids=["missing", "not_an_image", "directory"],
)
def test_unreadable_image_gets_an_error_and_others_are_analysed(
    service_session, two_colour_image, tmp_path, make_bad_path
):
    """A file that cannot be loaded gives its own result an error and no detections; the
    images around it are analysed normally."""
    bad = str(make_bad_path(tmp_path))
    good = str(two_colour_image)

    response = analyze_images([good, bad, good], service_session)

    assert [r.image_path for r in response.results] == [good, bad, good]
    failed = response.results[1]
    assert isinstance(failed.error, str) and failed.error
    assert failed.detections == []
    assert failed.detection_model == ModelInfo(**DETECTOR_INFO)
    assert failed.classification_model == ModelInfo(**CLASSIFIER_INFO)
    for result in (response.results[0], response.results[2]):
        assert result.error is None
        assert len(result.detections) == 2


def test_decompression_bomb_gets_an_error(service_session, two_colour_image, tmp_path, monkeypatch):
    """An image above Pillow's decompression-bomb limit is reported, not analysed."""
    bomb = tmp_path / "bomb.png"
    Image.new("RGB", (40, 40)).save(bomb)
    # 40x40 = 1600 pixels exceeds twice this limit (an error); the 16x8 image stays below it
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 300)

    response = analyze_images([str(bomb), str(two_colour_image)], service_session)

    assert response.results[0].error
    assert response.results[0].detections == []
    assert response.results[1].error is None


def test_all_images_unreadable(service_session, tmp_path):
    """If no image loads, every result carries an error and the response is still complete."""
    paths = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    response = analyze_images(paths, service_session)
    assert [r.image_path for r in response.results] == paths
    assert all(r.error and r.detections == [] for r in response.results)


def test_truncated_image_is_analysed(service_session, tmp_path):
    """A truncated JPEG is analysed as far as it decodes, without an error."""
    truncated = write_truncated_jpeg(tmp_path / "truncated.jpg")

    (result,) = analyze_images([str(truncated)], service_session).results

    assert result.error is None
    assert [d.detected_class for d in result.detections] == ["animal", "human"]


def test_model_errors_are_not_turned_into_image_errors(service_configs, two_colour_image, tmp_path):
    """Only loading errors are per-image results; a model that breaks its contract is a
    server fault and propagates (the API answers 500)."""
    _, classifier_config = service_configs
    session = Session.build(
        ModelConfig.from_file(write_broken_detector_config(tmp_path)), ModelConfig.from_file(classifier_config)
    )
    with pytest.raises(ValueError):
        analyze_images([str(two_colour_image)], session)
