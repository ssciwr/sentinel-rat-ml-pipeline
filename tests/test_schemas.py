"""Tests for the request/response schemas (`ml_pipeline.schemas`)."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ml_pipeline.schemas.analysis import Classification, Taxon
from ml_pipeline.schemas.session import ClassifierCard, DetectorCard, SessionInfo, SessionRequest

BROWN_RAT = {
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class_name": "Mammalia",
    "order": "Rodentia",
    "family": "Muridae",
    "genus": "Rattus",
    "species": "norvegicus",
    "common_name": "brown rat",
}


def test_taxon_genus_is_optional():
    """Family/order-level classes (and labels without taxonomy) have no genus."""
    assert Taxon(family="Sciuridae", common_name="sciuridae family").genus is None
    assert Taxon(common_name="blank").model_dump(exclude={"common_name"}) == dict.fromkeys(
        ["kingdom", "phylum", "class_name", "order", "family", "genus", "species"]
    )


def test_taxon_rejects_unknown_keys():
    """A misspelled taxonomy key in a config is an error, not silently dropped."""
    with pytest.raises(ValidationError):
        Taxon(genus="Rattus", specis="norvegicus")


def test_classification_json_is_unchanged_by_taxon_base():
    """Classification = Taxon + confidence; its serialised JSON keeps the README's keys,
    order and content."""
    classification = Classification(**Taxon(**BROWN_RAT).model_dump(), confidence=0.7)
    serialised = json.loads(classification.model_dump_json())
    assert list(serialised) == [*BROWN_RAT, "confidence"]
    assert serialised == BROWN_RAT | {"confidence": 0.7}


def test_classification_without_genus_serialises_null():
    """A family-level classification keeps every key; genus is null."""
    serialised = json.loads(Classification(family="Sciuridae", confidence=0.4).model_dump_json())
    assert list(serialised) == [*BROWN_RAT, "confidence"]
    assert serialised["genus"] is None


CARD_FIELDS = {
    "type": "SpeciesnetDetector",
    "info": {"name": "speciesnet-detector", "version": "v4.0.3b", "description": None},
    "source": "/etc/ml/detector.yaml",
    "conf_limit": 0.1,
    "weights": "kaggle:google/speciesnet/pyTorch/v4.0.3b/1",
}
DETECTOR_CARD = CARD_FIELDS | {"role": "detector", "labels": ["animal", "human", "vehicle"]}
CLASSIFIER_CARD = CARD_FIELDS | {"type": "SpeciesnetClassifier", "role": "classifier", "num_classes": 2498}


def test_session_info_json_shape():
    """A session is described by its creation time and one card per model: the detector card
    carries the labels, the classifier card the number of classes, and neither the other's field."""
    info = SessionInfo(
        created_at=datetime(2026, 11, 18, 12, 0, tzinfo=UTC),
        models=[DetectorCard(**DETECTOR_CARD), ClassifierCard(**CLASSIFIER_CARD)],
    )

    serialised = json.loads(info.model_dump_json())

    assert list(serialised) == ["created_at", "models"]
    assert serialised["created_at"].startswith("2026-11-18T12:00:00")
    assert serialised["models"] == [DETECTOR_CARD, CLASSIFIER_CARD]


def test_session_info_parses_each_card_by_its_role():
    """Read back from JSON, each card becomes the card type its role names."""
    info = SessionInfo.model_validate(
        {"created_at": "2026-11-18T12:00:00Z", "models": [DETECTOR_CARD, CLASSIFIER_CARD]}
    )
    assert [type(card) for card in info.models] == [DetectorCard, ClassifierCard]


@pytest.mark.parametrize(
    "card",
    [
        CARD_FIELDS | {"role": "segmenter", "labels": ["animal"]},  # no such role
        CARD_FIELDS | {"role": "detector"},  # a detector card needs its labels
        CARD_FIELDS | {"role": "classifier"},  # a classifier card needs its number of classes
        DETECTOR_CARD | {"num_classes": 3},  # a detector card has no number of classes
        CLASSIFIER_CARD | {"labels": ["animal"]},  # a classifier card has no labels
        CLASSIFIER_CARD | {"role": "detector"},  # role and fields must match
    ],
    ids=[
        "unknown_role",
        "detector_without_labels",
        "classifier_without_count",
        "detector_with_count",
        "classifier_with_labels",
        "detector_role_with_classifier_fields",
    ],
)
def test_invalid_model_cards_are_rejected(card):
    """A card's role is one of the two pipeline roles, and it carries exactly that role's fields."""
    with pytest.raises(ValidationError):
        SessionInfo.model_validate({"created_at": "2026-11-18T12:00:00Z", "models": [card]})


@pytest.mark.parametrize(
    ("card_type", "card"),
    [(DetectorCard, DETECTOR_CARD | {"role": "classifier"}), (ClassifierCard, CLASSIFIER_CARD | {"role": "detector"})],
)
def test_model_card_role_is_fixed_by_its_type(card_type, card):
    """A detector card cannot claim the classifier role, nor the reverse."""
    with pytest.raises(ValidationError):
        card_type(**card)


def test_session_request_configs_are_optional_mappings():
    """Either role's config may be omitted; a given one must be a mapping."""
    assert SessionRequest().detector is None
    assert SessionRequest(classifier={"type": "TorchClassifier"}).classifier == {"type": "TorchClassifier"}
    with pytest.raises(ValidationError):
        SessionRequest(detector=["type", "TorchDetector"])


@pytest.mark.parametrize("key", ["detectr", "Detector", "classifiers"])
def test_session_request_rejects_unknown_roles(key):
    """A misspelled role is an error, not silently ignored (which would rebuild it from env)."""
    with pytest.raises(ValidationError):
        SessionRequest(**{key: {"type": "TorchDetector"}})
