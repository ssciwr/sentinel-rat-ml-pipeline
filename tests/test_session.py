"""Tests for the session (`ml_pipeline.services.session`), run with the tiny service models."""

from datetime import UTC, datetime

import pytest
import yaml

from ml_pipeline.models.model import TorchClassifier, TorchDetector
from ml_pipeline.schemas.analysis import ModelInfo
from ml_pipeline.schemas.session import ClassifierCard, DetectorCard, SessionInfo
from ml_pipeline.services.session import ModelConfig, Session
from tests.service_support import CLASSIFIER_INFO, COLOUR_TAXA, DETECTOR_INFO, DETECTOR_LABELS


def _config_values(path) -> dict:
    """The mapping of a YAML config file."""
    return yaml.safe_load(path.read_text())


def test_model_config_from_file_keeps_values_and_path(service_configs):
    """A config read from a file keeps its parsed mapping and names the file as its source."""
    detector_config, _ = service_configs
    config = ModelConfig.from_file(detector_config)
    assert config.values == _config_values(detector_config)
    assert config.source == str(detector_config)


def test_model_config_from_request_keeps_values(service_configs):
    """An uploaded config keeps its mapping; its source is the request."""
    detector_config, _ = service_configs
    values = _config_values(detector_config)
    config = ModelConfig.from_request(values)
    assert config.values == values
    assert config.source == "request"


def test_model_config_from_missing_file_raises(tmp_path):
    """A missing config file is a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ModelConfig.from_file(tmp_path / "missing.yaml")


def test_session_holds_models_and_configs(service_session, service_configs):
    """A built session holds both models, the configs they were built from and its UTC
    creation time."""
    detector_config, classifier_config = service_configs
    assert type(service_session.detector) is TorchDetector
    assert type(service_session.classifier) is TorchClassifier
    assert service_session.detector_config.source == str(detector_config)
    assert service_session.classifier_config.source == str(classifier_config)
    assert isinstance(service_session.created_at, datetime)
    assert service_session.created_at.utcoffset() == UTC.utcoffset(None)


@pytest.mark.parametrize(
    ("detector_role", "classifier_role"),
    [("detector", "detector"), ("classifier", "classifier"), ("classifier", "detector")],
    ids=["detector_as_classifier", "classifier_as_detector", "swapped"],
)
def test_session_rejects_a_model_of_the_wrong_kind(service_configs, detector_role, classifier_role):
    """Each role must get a model of its kind: a detector config as classifier, a classifier
    config as detector, or both swapped are TypeErrors."""
    paths = dict(zip(["detector", "classifier"], service_configs))
    with pytest.raises(TypeError):
        Session.build(ModelConfig.from_file(paths[detector_role]), ModelConfig.from_file(paths[classifier_role]))


def test_wrong_kind_is_rejected_before_any_model_is_loaded(service_configs, tmp_path):
    """A config of the wrong kind fails on its `type`, before the other config's (here missing)
    weights would be loaded."""
    detector_config, _ = service_configs
    classifier_values = _config_values(detector_config) | {"model_path": str(tmp_path / "missing.pt")}
    with pytest.raises(TypeError):
        Session.build(ModelConfig.from_file(detector_config), ModelConfig.from_request(classifier_values))


def test_session_info_has_one_card_per_model(service_session, service_configs):
    """The model cards describe the running detector and classifier, detector first."""
    detector_config, classifier_config = service_configs

    info = service_session.info()

    assert isinstance(info, SessionInfo)
    assert info.created_at == service_session.created_at
    detector_card, classifier_card = info.models
    assert detector_card == DetectorCard(
        role="detector",
        type="TorchDetector",
        info=ModelInfo(**DETECTOR_INFO),
        source=str(detector_config),
        conf_limit=0.5,
        weights=_config_values(detector_config)["model_path"],
        labels=DETECTOR_LABELS,
    )
    assert classifier_card == ClassifierCard(
        role="classifier",
        type="TorchClassifier",
        info=ModelInfo(**CLASSIFIER_INFO),
        source=str(classifier_config),
        conf_limit=0.05,
        weights=_config_values(classifier_config)["model_path"],
        num_classes=len(COLOUR_TAXA),
    )


def test_detector_card_shows_labels_after_label_map(service_configs):
    """The card lists the labels the detector returns, i.e. with label_map applied."""
    detector_config, classifier_config = service_configs
    values = _config_values(detector_config) | {"label_map": {"human": "person"}}

    session = Session.build(ModelConfig.from_request(values), ModelConfig.from_file(classifier_config))

    detector_card = session.info().models[0]
    assert detector_card.labels == ["animal", "person"]
    assert detector_card.source == "request"


def test_detector_card_lists_merged_labels_once(service_configs):
    """If label_map merges labels, each returned label is listed once."""
    detector_config, classifier_config = service_configs
    values = _config_values(detector_config) | {"label_map": {"human": "animal"}}

    session = Session.build(ModelConfig.from_request(values), ModelConfig.from_file(classifier_config))

    assert session.info().models[0].labels == ["animal"]
