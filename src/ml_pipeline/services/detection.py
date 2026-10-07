"""Animal detection service."""

from ml_pipeline.schemas.analysis import BoundingBox, Detection, ModelInfo

DETECTION_MODEL = ModelInfo(
    name="megadetector",
    version="v5a",
    description="Simulated detector (animal, person, vehicle)",
)


def detect_animals(image_path: str) -> list[Detection]:
    """Detect objects in one image, without species classifications."""
    # TODO: replace with real model inference
    return [
        Detection(
            detected_class="animal",
            confidence=0.92,
            bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.5),
        ),
        Detection(
            detected_class="person",
            confidence=0.81,
            bbox=BoundingBox(x_min=0.6, y_min=0.1, x_max=0.9, y_max=0.95),
        ),
    ]
