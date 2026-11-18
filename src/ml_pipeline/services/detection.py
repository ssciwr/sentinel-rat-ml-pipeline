"""Detection service: runs a detector on one image."""

from PIL import Image

from ml_pipeline.models.model import DetectorBase
from ml_pipeline.schemas.analysis import Detection


def detect_animals(detector: DetectorBase, image: Image.Image) -> list[Detection]:
    """Detect objects in one image.

    Args:
        detector: The detector to run (the session's).
        image: An RGB image.

    Returns:
        All detections above the detector's confidence limit, most confident first, each
        labelled (`detected_class`) and without species classifications.
    """
    _, detections = detector.predict(image)
    return detections
