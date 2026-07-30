"""Animal detection service."""

from ml_pipeline.schemas.analysis import AnalyzeResponse

SIMULATED_SPECIES = {"cat": 1, "rodent": 1}


def detect_animals(image_path: str) -> AnalyzeResponse:
    # TODO: replace with real model inference
    return AnalyzeResponse(
        image_path=image_path,
        animals_detected=2,
        species=SIMULATED_SPECIES,
        confidence=0.95,
    )
