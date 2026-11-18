"""Tests for the request/response schemas (`ml_pipeline.schemas.analysis`)."""

import json

import pytest
from pydantic import ValidationError

from ml_pipeline.schemas.analysis import Classification, Taxon

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


def test_taxon_requires_genus():
    """Taxonomy entries are validated; genus is mandatory."""
    with pytest.raises(ValidationError):
        Taxon(species="rattus")


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
