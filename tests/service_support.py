"""Tiny-model setup shared by the service and API tests.

The detector always returns two boxes, an `animal` box over the left half of the image
and a `human` box over the right half. The classifier classifies by dominant colour. On
the two-colour test image (left half red, right half blue) a correct crop of the animal
box therefore classifies as the red taxon only, while the whole image would split
between red and blue.
"""

import io
from pathlib import Path

import torch
import yaml
from PIL import Image

from tests import tiny_models

DETECTOR_INFO = {"name": "tiny-detector", "version": "1", "description": "fixed animal and human boxes"}
CLASSIFIER_INFO = {"name": "tiny-classifier", "version": "2", "description": "dominant colour"}

DETECTOR_LABELS = ["animal", "human"]
ANIMAL_BOX = {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0}
HUMAN_BOX = {"x_min": 0.5, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
DETECTOR_ROWS = [
    [*ANIMAL_BOX.values(), 0.9, 0.0],
    [*HUMAN_BOX.values(), 0.8, 1.0],
]

COLOUR_TAXA = [
    {"genus": "Rattus", "species": "rubrum", "common_name": "red rat"},
    {"genus": "Mus", "species": "viridis", "common_name": "green mouse"},
    {"genus": "Microtus", "species": "caeruleus", "common_name": "blue vole"},
]
RED_TAXON = COLOUR_TAXA[0]

# Size of the two-colour image; small enough to stay below lowered decompression-bomb limits.
IMAGE_SIZE = (16, 8)


def write_service_configs(directory: Path) -> tuple[Path, Path]:
    """Pickle the tiny detector and classifier and write their YAML configs.

    Args:
        directory: Where the `.pt` files and configs are written.

    Returns:
        The detector config path and the classifier config path.
    """
    detector_pt = directory / "service_detector.pt"
    classifier_pt = directory / "service_classifier.pt"
    torch.save(tiny_models.FixedRowsDetector(DETECTOR_ROWS), detector_pt)
    torch.save(tiny_models.ColourClassifier(), classifier_pt)

    detector_config = _write_detector_config(directory / "detector.yaml", detector_pt)
    classifier_config = directory / "classifier.yaml"
    classifier_config.write_text(
        yaml.safe_dump(
            {
                "type": "TorchClassifier",
                "info": CLASSIFIER_INFO,
                "model_path": str(classifier_pt),
                "conf_limit": 0.05,
                "preprocessing": {},
                "classes": COLOUR_TAXA,
                "apply_softmax": True,
            }
        )
    )
    return detector_config, classifier_config


def write_broken_detector_config(directory: Path) -> Path:
    """Write a detector config whose model breaks the output contract (a score above 1).

    Args:
        directory: Where the `.pt` file and the config are written.

    Returns:
        The config path.
    """
    model_pt = directory / "broken_detector.pt"
    torch.save(tiny_models.ContractBreaker("det_score_range"), model_pt)
    return _write_detector_config(directory / "broken_detector.yaml", model_pt)


def _write_detector_config(path: Path, model_pt: Path) -> Path:
    """Write the service's TorchDetector config for the pickled module `model_pt`."""
    path.write_text(
        yaml.safe_dump(
            {
                "type": "TorchDetector",
                "info": DETECTOR_INFO,
                "model_path": str(model_pt),
                "conf_limit": 0.5,
                "preprocessing": {},
                "labels": DETECTOR_LABELS,
            }
        )
    )
    return path


def write_truncated_jpeg(path: Path) -> Path:
    """Save a noisy 64x48 JPEG cut off halfway through its image data.

    Noise keeps the file large, so the cut truncates image data rather than the header.

    Args:
        path: Target file.

    Returns:
        `path`.
    """
    buffer = io.BytesIO()
    Image.effect_noise((64, 48), 50).convert("RGB").save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    path.write_bytes(data[: len(data) // 2])
    return path


def write_non_image(path: Path) -> Path:
    """Write a text file (typically with an image suffix).

    Args:
        path: Target file.

    Returns:
        `path`.
    """
    path.write_text("not an image")
    return path


def write_two_colour_image(path: Path) -> Path:
    """Save an RGB image whose left half is red and right half is blue.

    Args:
        path: Target file; the suffix selects the format.

    Returns:
        `path`.
    """
    width, height = IMAGE_SIZE
    image = Image.new("RGB", IMAGE_SIZE, (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, width // 2, height))
    image.save(path)
    return path
