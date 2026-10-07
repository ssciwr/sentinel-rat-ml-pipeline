"""API endpoints."""

from fastapi import APIRouter, status

from ml_pipeline.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from ml_pipeline.services.analysis import analyze_images

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    return analyze_images(payload.image_paths)
