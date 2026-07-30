"""Tests for ML Pipeline configuration."""

from ml_pipeline.config import Settings


def test_default_settings() -> None:
    settings = Settings(ML_PIPELINE_PORT=8000)
    assert settings.ML_PIPELINE_PORT == 8000
