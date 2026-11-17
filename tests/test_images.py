"""Tests for reading the requested images from the watch folder."""

from pathlib import Path

import pytest

from ml_pipeline.services.images import image_file, read_image


def test_read_image_from_watch_folder(watch_folder: Path) -> None:
    assert image_file("hd-zoo-01/a.jpg") == (watch_folder / "hd-zoo-01/a.jpg").resolve()
    assert read_image("hd-zoo-01/a.jpg") == b"fake jpeg"


@pytest.mark.parametrize(
    "image_path", ["/etc/passwd", "../outside.jpg", "x/../../y.jpg"]
)
@pytest.mark.usefixtures("watch_folder")
def test_read_image_outside_watch_folder(image_path: str) -> None:
    with pytest.raises(ValueError):
        read_image(image_path)


def test_read_image_symlink_outside_watch_folder(
    watch_folder: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # a link in the watch folder may not expose a file outside of it
    outside = tmp_path_factory.mktemp("outside") / "secret.jpg"
    outside.write_bytes(b"secret")
    (watch_folder / "hd-zoo-01" / "link.jpg").symlink_to(outside)

    with pytest.raises(ValueError):
        read_image("hd-zoo-01/link.jpg")


@pytest.mark.usefixtures("watch_folder")
def test_read_image_missing() -> None:
    with pytest.raises(OSError):
        read_image("hd-zoo-01/missing.jpg")
