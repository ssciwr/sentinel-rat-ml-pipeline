"""Schemas for ML pipeline requests and responses."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_paths: list[str] = Field(min_length=1)


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


class Classification(BaseModel):
    kingdom: str | None = None
    phylum: str | None = None
    class_name: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str
    species: str | None = None
    common_name: str | None = None
    confidence: float


class Detection(BaseModel):
    detected_class: str
    confidence: float
    bbox: BoundingBox
    classifications: list[Classification] = []


class ImageResult(BaseModel):
    image_path: str
    detection_model: ModelInfo
    classification_model: ModelInfo
    detections: list[Detection]


class AnalyzeResponse(BaseModel):
    """One result per requested image, in request order."""

    results: list[ImageResult]
