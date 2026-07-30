"""API endpoints."""

from fastapi import APIRouter, status

from ml_pipeline.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from ml_pipeline.services.detection import detect_animals

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_image(payload: AnalyzeRequest) -> AnalyzeResponse:
    return detect_animals(payload.image_path)
