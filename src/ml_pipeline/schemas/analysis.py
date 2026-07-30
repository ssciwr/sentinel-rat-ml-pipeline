"""Schemas for ML pipeline requests and responses."""

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    image_path: str


class AnalyzeResponse(BaseModel):
    image_path: str
    animals_detected: int
    species: dict[str, int]
    confidence: float
