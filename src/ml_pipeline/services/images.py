"""Load requested images and cut detection crops for the classifier."""

import math
from pathlib import Path

from PIL import Image, ImageOps

from ml_pipeline.schemas.analysis import BoundingBox


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB, with its EXIF orientation applied.

    The RGB conversion decodes the whole file, so files that are not images fail here. Truncated
    images load as far as they decode: importing the SpeciesNet SDK (done by the model module)
    sets `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True` process-wide, and the pipeline accepts that.

    Args:
        path: Path of the image file.

    Returns:
        The decoded RGB image.

    Raises:
        OSError: The file is missing, unreadable, a directory or not an image
            (`FileNotFoundError`, `PIL.UnidentifiedImageError`, ...).
        PIL.Image.DecompressionBombError: The image exceeds Pillow's pixel limit.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop(image: Image.Image, bbox: BoundingBox) -> Image.Image:
    """Cut the region of a relative bounding box out of an image.

    The box is widened outward to whole pixels and clamped to the image; a zero-area box still
    yields one pixel. No margin is added.

    Args:
        image: The full image.
        bbox: Box with corners relative to the image size (0-1).

    Returns:
        The cropped region, at least 1x1 pixels.
    """
    width, height = image.size
    left, right = _pixel_span(bbox.x_min, bbox.x_max, width)
    top, bottom = _pixel_span(bbox.y_min, bbox.y_max, height)
    return image.crop((left, top, right, bottom))


def _pixel_span(low: float, high: float, size: int) -> tuple[int, int]:
    """Map a relative [low, high] span to pixels [start, end): floor/ceil outward, clamped to
    [0, size], at least one pixel wide."""
    start = min(max(math.floor(low * size), 0), size - 1)
    end = min(max(math.ceil(high * size), start + 1), size)
    return start, end
