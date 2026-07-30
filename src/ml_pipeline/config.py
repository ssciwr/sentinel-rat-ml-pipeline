"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ML_PIPELINE_PORT: int = 8000

    class Config:
        env_prefix = ""


settings = Settings()
