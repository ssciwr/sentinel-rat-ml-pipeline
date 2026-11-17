"""Shared fixtures."""

from pathlib import Path

import pytest

from ml_pipeline.config import settings

IMAGES = ("hd-zoo-01/a.jpg", "hd-zoo-01/b.jpg")


@pytest.fixture
def watch_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A watch folder with the images in IMAGES, used as WATCH_FOLDER_DOCKER."""
    for image in IMAGES:
        path = tmp_path / image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpeg")
    monkeypatch.setattr(settings, "WATCH_FOLDER_DOCKER", str(tmp_path))
    return tmp_path
