"""Schemas for ML pipeline requests and responses."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    image_paths: list[str] = Field(min_length=1)

    @field_validator("image_paths")
    @classmethod
    def _validate_paths(cls, image_paths: list[str]) -> list[str]:
        """Reject relative paths (the service has no base directory to resolve them against) and
        paths with NUL bytes (no file system accepts them; opening one raises ValueError)."""
        relative = [path for path in image_paths if not Path(path).is_absolute()]
        if relative:
            raise ValueError(f"image paths must be absolute, got {relative}")
        if any("\x00" in path for path in image_paths):
            raise ValueError("image paths must not contain NUL bytes")
        return image_paths


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    description: str | None = None


class BoundingBox(BaseModel):
    """Bounding box in coordinates relative to the image size (0-1)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Taxon(BaseModel):
    """Taxonomy of one species class, as configured for a classifier's output index."""

    # Taxonomy comes from config files: a misspelled key must be an error, not silently dropped.
    model_config = ConfigDict(extra="forbid")

    kingdom: str | None = None
    phylum: str | None = None
    class_name: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str
    species: str | None = None
    common_name: str | None = None


class Classification(Taxon):
    """A taxon with the classifier's confidence for it."""

    confidence: float


class Detection(BaseModel):
    detected_class: str
    confidence: float
    bbox: BoundingBox
    classifications: list[Classification] = []


class ImageResult(BaseModel):
    """Analysis result of one requested image."""

    image_path: str
    detection_model: ModelInfo
    classification_model: ModelInfo
    detections: list[Detection]
    # set (with empty detections) when the image could not be loaded
    error: str | None = None


class AnalyzeResponse(BaseModel):
    """One result per requested image, in request order."""

    results: list[ImageResult]
