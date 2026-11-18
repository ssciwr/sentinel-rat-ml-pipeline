"""Species classification service: runs a classifier on one crop."""

from PIL import Image

from ml_pipeline.models.model import ClassifierBase
from ml_pipeline.schemas.analysis import Classification


def classify_species(classifier: ClassifierBase, crop: Image.Image) -> list[Classification]:
    """Classify the species of one detected animal.

    Args:
        classifier: The classifier to run (the session's).
        crop: The RGB image region of the detection.

    Returns:
        Species candidates above the classifier's confidence limit, most likely first.
    """
    _, classifications = classifier.predict(crop)
    return classifications
