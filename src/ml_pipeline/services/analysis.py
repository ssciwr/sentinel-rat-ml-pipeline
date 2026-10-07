"""Run detection and species classification on a batch of images."""

from ml_pipeline.schemas.analysis import AnalyzeResponse, ImageResult
from ml_pipeline.services.classification import CLASSIFICATION_MODEL, classify_species
from ml_pipeline.services.detection import DETECTION_MODEL, detect_animals

# only detections of this class are passed to species classification
CLASSIFIED_CLASS = "animal"


def analyze_images(image_paths: list[str]) -> AnalyzeResponse:
    results = []
    for image_path in image_paths:
        detections = detect_animals(image_path)
        for detection in detections:
            if detection.detected_class == CLASSIFIED_CLASS:
                detection.classifications = classify_species(image_path, detection)
        results.append(
            ImageResult(
                image_path=image_path,
                detection_model=DETECTION_MODEL,
                classification_model=CLASSIFICATION_MODEL,
                detections=detections,
            )
        )
    return AnalyzeResponse(results=results)
