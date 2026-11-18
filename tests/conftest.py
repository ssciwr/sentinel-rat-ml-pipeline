"""Fixtures shared by the service, session and API tests."""

from pathlib import Path

import pytest

from ml_pipeline.services.session import ModelConfig, Session
from tests.service_support import write_service_configs


@pytest.fixture(scope="session")
def service_configs(tmp_path_factory) -> tuple[Path, Path]:
    """Detector and classifier configs of the tiny service models (see tests/service_support.py)."""
    return write_service_configs(tmp_path_factory.mktemp("service_models"))


@pytest.fixture
def service_session(service_configs) -> Session:
    """A session running the tiny service models, built from their config files."""
    detector_config, classifier_config = service_configs
    return Session.build(ModelConfig.from_file(detector_config), ModelConfig.from_file(classifier_config))
