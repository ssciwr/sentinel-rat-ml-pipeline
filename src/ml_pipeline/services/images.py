"""Read the requested images from the watch folder."""

from pathlib import Path

from ml_pipeline.config import settings


def image_file(image_path: str) -> Path:
    """Path of an image in the watch folder.

    ``image_path`` is relative to the watch folder (WATCH_FOLDER_DOCKER), e.g.
    "hd-zoo-01/image_20260730T115638Z.jpg". A path that is absolute or leads
    outside the watch folder raises ValueError.
    """
    root = Path(settings.WATCH_FOLDER_DOCKER).resolve()
    if Path(image_path).is_absolute():
        raise ValueError(f"{image_path} is not relative to the watch folder")
    path = (root / image_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{image_path} is outside the watch folder")
    return path


def read_image(image_path: str) -> bytes:
    """Content of an image in the watch folder. Raises ValueError for a path
    outside the watch folder and OSError if the file can't be read."""
    return image_file(image_path).read_bytes()
