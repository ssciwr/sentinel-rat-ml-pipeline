"""Species classification service: runs the configured classifier on one crop."""

import functools

from PIL import Image

from ml_pipeline.config import settings
from ml_pipeline.models.model import ClassifierBase, load_model
from ml_pipeline.schemas.analysis import Classification, ModelInfo


def classify_species(crop: Image.Image) -> list[Classification]:
    """Classify the species of one detected animal.

    Args:
        crop: The RGB image region of the detection.

    Returns:
        Species candidates above the classifier's confidence limit, most likely first.

    Raises:
        RuntimeError: No classifier is configured (`CLASSIFIER_CONFIG` is unset).
        TypeError: `CLASSIFIER_CONFIG` configures a model that is not a classifier.
    """
    _, classifications = _classifier().predict(crop)
    return classifications


def classification_model() -> ModelInfo:
    """Name, version and description of the configured classifier.

    Raises:
        RuntimeError: No classifier is configured (`CLASSIFIER_CONFIG` is unset).
        TypeError: `CLASSIFIER_CONFIG` configures a model that is not a classifier.
    """
    return _classifier().info


@functools.cache
def _classifier() -> ClassifierBase:
    """Build the classifier from `settings.CLASSIFIER_CONFIG` on first use, then reuse it.

    The model is cached per process for now; a later change replaces this with a session object
    that can load and reload models on request.
    """
    if settings.CLASSIFIER_CONFIG is None:
        raise RuntimeError(
            "no classifier configured: set the CLASSIFIER_CONFIG environment variable to a model config file"
        )
    model = load_model(settings.CLASSIFIER_CONFIG)
    if not isinstance(model, ClassifierBase):
        raise TypeError(f"CLASSIFIER_CONFIG must configure a classifier, got {type(model).__name__}")
    return model
