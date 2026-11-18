"""Tests for ML Pipeline configuration."""

from pathlib import Path

import pytest

from ml_pipeline.config import Settings


@pytest.fixture
def clean_env(monkeypatch):
    """Remove the settings' environment variables, so only each test's values apply."""
    for name in ("DETECTOR_CONFIG", "CLASSIFIER_CONFIG", "ML_PIPELINE_PORT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_settings(clean_env) -> None:
    """The port keeps its default; the model configs are unset unless configured."""
    settings = Settings(_env_file=None)

    assert settings.ML_PIPELINE_PORT == 8000
    assert settings.DETECTOR_CONFIG is None
    assert settings.CLASSIFIER_CONFIG is None


def test_model_configs_are_read_from_env_vars(clean_env) -> None:
    """DETECTOR_CONFIG and CLASSIFIER_CONFIG are set through environment variables."""
    clean_env.setenv("DETECTOR_CONFIG", "/etc/ml/detector.yaml")
    clean_env.setenv("CLASSIFIER_CONFIG", "/etc/ml/classifier.yaml")

    settings = Settings(_env_file=None)

    assert settings.DETECTOR_CONFIG == Path("/etc/ml/detector.yaml")
    assert settings.CLASSIFIER_CONFIG == Path("/etc/ml/classifier.yaml")


def test_empty_env_var_counts_as_unset(clean_env) -> None:
    """`DETECTOR_CONFIG=` (as in .env.example) means unset, not the current directory."""
    clean_env.setenv("DETECTOR_CONFIG", "")
    clean_env.setenv("CLASSIFIER_CONFIG", "")

    settings = Settings(_env_file=None)

    assert settings.DETECTOR_CONFIG is None
    assert settings.CLASSIFIER_CONFIG is None
