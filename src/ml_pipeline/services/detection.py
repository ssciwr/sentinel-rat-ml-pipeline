"""Detection service: runs the configured detector on one image."""

import functools

from PIL import Image

from ml_pipeline.config import settings
from ml_pipeline.models.model import DetectorBase, load_model
from ml_pipeline.schemas.analysis import Detection, ModelInfo


def detect_animals(image: Image.Image) -> list[Detection]:
    """Detect objects in one image.

    Args:
        image: An RGB image.

    Returns:
        All detections above the detector's confidence limit, most confident first, each
        labelled (`detected_class`) and without species classifications.

    Raises:
        RuntimeError: No detector is configured (`DETECTOR_CONFIG` is unset).
        TypeError: `DETECTOR_CONFIG` configures a model that is not a detector.
    """
    _, detections = _detector().predict(image)
    return detections


def detection_model() -> ModelInfo:
    """Name, version and description of the configured detector.

    Raises:
        RuntimeError: No detector is configured (`DETECTOR_CONFIG` is unset).
        TypeError: `DETECTOR_CONFIG` configures a model that is not a detector.
    """
    return _detector().info


@functools.cache
def _detector() -> DetectorBase:
    """Build the detector from `settings.DETECTOR_CONFIG` on first use, then reuse it.

    The model is cached per process for now; a later change replaces this with a session object
    that can load and reload models on request.
    """
    if settings.DETECTOR_CONFIG is None:
        raise RuntimeError("no detector configured: set the DETECTOR_CONFIG environment variable to a model config file")
    model = load_model(settings.DETECTOR_CONFIG)
    if not isinstance(model, DetectorBase):
        raise TypeError(f"DETECTOR_CONFIG must configure a detector, got {type(model).__name__}")
    return model
