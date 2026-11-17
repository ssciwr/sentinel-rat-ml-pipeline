"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    ML_PIPELINE_PORT: int = 8000
    # watch folder mounted into this container (read-only); requested image paths
    # are relative to it, as sent by sentinel-rat-pipeline
    WATCH_FOLDER_DOCKER: str = "/data/images"


settings = Settings()
