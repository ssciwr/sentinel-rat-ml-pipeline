"""Run detection and species classification on a batch of images."""

import logging

from ml_pipeline.schemas.analysis import AnalyzeResponse, ImageResult
from ml_pipeline.services.classification import CLASSIFICATION_MODEL, classify_species
from ml_pipeline.services.detection import DETECTION_MODEL, detect_animals
from ml_pipeline.services.images import read_image

logger = logging.getLogger(__name__)

# only detections of this class are passed to species classification
CLASSIFIED_CLASS = "animal"


def analyze_images(image_paths: list[str]) -> AnalyzeResponse:
    """Analyze each image, read from the watch folder.

    An image that can't be read (missing, unreadable, or outside the watch folder)
    is left out of the results and logged, so the rest of the batch is still
    analyzed; sentinel-rat-pipeline then skips it and retries at its next start.
    """
    results = []
    for image_path in image_paths:
        try:
            image = read_image(image_path)
        except (OSError, ValueError) as exc:
            logger.error("Skipping %s: %s", image_path, exc)
            continue

        detections = detect_animals(image)
        for detection in detections:
            if detection.detected_class == CLASSIFIED_CLASS:
                detection.classifications = classify_species(image, detection)
        results.append(
            ImageResult(
                image_path=image_path,
                detection_model=DETECTION_MODEL,
                classification_model=CLASSIFICATION_MODEL,
                detections=detections,
            )
        )
    return AnalyzeResponse(results=results)
