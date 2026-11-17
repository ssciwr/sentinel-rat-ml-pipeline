"""Tests for ML Pipeline configuration."""

from ml_pipeline.config import Settings


def test_default_settings() -> None:
    settings = Settings(ML_PIPELINE_PORT=8000)
    assert settings.ML_PIPELINE_PORT == 8000
    assert settings.WATCH_FOLDER_DOCKER == "/data/images"


def test_watch_folder_from_env(monkeypatch) -> None:
    monkeypatch.setenv("WATCH_FOLDER_DOCKER", "/mnt/images")
    assert Settings().WATCH_FOLDER_DOCKER == "/mnt/images"
