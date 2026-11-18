"""Run detection and species classification on a list of images, one at a time."""

from PIL import Image

from ml_pipeline.schemas.analysis import AnalyzeResponse, Detection, ImageResult
from ml_pipeline.services.classification import classify_species
from ml_pipeline.services.detection import detect_animals
from ml_pipeline.services.images import crop, load_image
from ml_pipeline.services.session import Session

# only detections of this class are passed to species classification
CLASSIFIED_CLASS = "animal"


def analyze_images(image_paths: list[str], session: Session) -> AnalyzeResponse:
    """Analyse each image: detect objects, then classify the species of every animal.

    Args:
        image_paths: Absolute image paths.
        session: The session whose detector and classifier run.

    Returns:
        One result per path, in request order. An image that cannot be loaded gets an `error`
        and no detections; the other images are analysed normally.
    """
    return AnalyzeResponse(results=[_analyze_image(image_path, session) for image_path in image_paths])


def _analyze_image(image_path: str, session: Session) -> ImageResult:
    """Load one image, detect objects in it and classify the crop of each animal detection.

    Only loading errors are caught (they belong to this image); model errors are server faults
    and propagate.
    """
    try:
        image = load_image(image_path)
    except (OSError, Image.DecompressionBombError) as error:
        # anticipated: the file is missing, unreadable, not an image or too large
        detections, error_message = [], f"{type(error).__name__}: {error}"
    else:
        # outside the try: model errors are not this image's loading error
        detections, error_message = _detect_and_classify(image, session), None
    return ImageResult(
        image_path=image_path,
        detection_model=session.detector.info,
        classification_model=session.classifier.info,
        detections=detections,
        error=error_message,
    )


def _detect_and_classify(image: Image.Image, session: Session) -> list[Detection]:
    """Detect objects in a loaded image and classify the crop of each animal detection."""
    detections = detect_animals(session.detector, image)
    for detection in detections:
        if detection.detected_class == CLASSIFIED_CLASS:
            detection.classifications = classify_species(session.classifier, crop(image, detection.bbox))
    return detections
