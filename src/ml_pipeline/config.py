"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, read from environment variables."""

    # env_ignore_empty: `DETECTOR_CONFIG=` means unset, not the current directory
    model_config = SettingsConfigDict(env_prefix="", env_ignore_empty=True)

    ML_PIPELINE_PORT: int = 8000
    DETECTOR_CONFIG: Path | None = None
    CLASSIFIER_CONFIG: Path | None = None


settings = Settings()
