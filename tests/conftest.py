"""Fixtures shared by the service and API tests."""

from pathlib import Path

import pytest

from tests.service_support import write_service_configs


@pytest.fixture(scope="session")
def service_configs(tmp_path_factory) -> tuple[Path, Path]:
    """Detector and classifier configs of the tiny service models (see tests/service_support.py)."""
    return write_service_configs(tmp_path_factory.mktemp("service_models"))


def _clear_model_caches() -> None:
    """Forget the models the services built, so the next call reads the current settings."""
    from ml_pipeline.services import classification, detection

    detection._detector.cache_clear()
    classification._classifier.cache_clear()


@pytest.fixture
def configured_services(monkeypatch, service_configs) -> tuple[Path, Path]:
    """Point the service settings at the tiny-model configs, with freshly built models."""
    from ml_pipeline.config import settings

    detector_config, classifier_config = service_configs
    monkeypatch.setattr(settings, "DETECTOR_CONFIG", detector_config)
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", classifier_config)
    _clear_model_caches()
    yield service_configs
    _clear_model_caches()
