"""Tests for image loading and cropping in the services (`ml_pipeline.services.images`)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from ml_pipeline.schemas.analysis import BoundingBox
from ml_pipeline.services.images import crop, load_image
from tests.service_support import write_non_image, write_truncated_jpeg

PROPERTY_SETTINGS = settings(deadline=None, max_examples=100)


# ------------------------------------------------------------------------- load_image


def test_load_image_applies_exif_orientation(tmp_path):
    """An EXIF 'rotate 90' tag is applied: width and height come back swapped."""
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW
    Image.new("RGB", (40, 20), (200, 10, 10)).save(path, exif=exif)
    assert load_image(path).size == (20, 40)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_load_image_converts_to_rgb(tmp_path, mode):
    """Grayscale, alpha and palette images come back as RGB."""
    path = tmp_path / f"image_{mode}.png"
    Image.new(mode, (8, 8)).save(path)
    assert load_image(path).mode == "RGB"


def test_load_image_accepts_str_paths(tmp_path):
    """Request paths arrive as strings."""
    path = tmp_path / "image.png"
    Image.new("RGB", (5, 3)).save(path)
    assert load_image(str(path)).size == (5, 3)


def test_load_image_missing_file_raises(tmp_path):
    """A missing file is an OSError (FileNotFoundError), for the service to report."""
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")


def test_load_image_non_image_file_raises(tmp_path):
    """A file that is not an image is an OSError, for the service to report."""
    with pytest.raises(OSError):
        load_image(write_non_image(tmp_path / "notes.jpg"))


def test_load_image_accepts_truncated_jpeg(tmp_path):
    """Truncated images load at their full size (the SpeciesNet SDK enables this in Pillow
    process-wide), so a partly written camera-trap image is still analysed."""
    import ml_pipeline.models.model  # noqa: F401  (imports the SDK, as the running service does)

    image = load_image(write_truncated_jpeg(tmp_path / "truncated.jpg"))
    assert image.size == (64, 48)
    assert image.mode == "RGB"


# ------------------------------------------------------------------------------- crop


def _coordinate_image(width: int, height: int) -> Image.Image:
    """An image whose pixel (x, y) has the colour (x + 1, y + 1, 255), so a crop reveals its
    origin. No pixel is black, unlike the padding PIL adds for crops beyond the image."""
    image = Image.new("RGB", (width, height))
    image.putdata([(x + 1, y + 1, 255) for y in range(height) for x in range(width)])
    return image


def _origin(cropped: Image.Image) -> tuple[int, int]:
    """Image coordinates of a crop's top-left pixel."""
    red, green, _ = cropped.getpixel((0, 0))
    return red - 1, green - 1


def _has_padding(cropped: Image.Image) -> bool:
    """True if the crop contains pixels from outside the image (PIL pads them black)."""
    return bool((np.asarray(cropped)[..., 2] != 255).any())


@st.composite
def images_and_boxes(draw):
    """An image size up to 64x64 and a valid relative box (min <= max, inside [0, 1])."""
    width = draw(st.integers(min_value=1, max_value=64))
    height = draw(st.integers(min_value=1, max_value=64))
    x_min, x_max = sorted(draw(st.lists(st.floats(0.0, 1.0), min_size=2, max_size=2)))
    y_min, y_max = sorted(draw(st.lists(st.floats(0.0, 1.0), min_size=2, max_size=2)))
    return width, height, BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


@PROPERTY_SETTINGS
@given(case=images_and_boxes())
def test_crop_covers_the_box_tightly_and_stays_inside(case):
    """The crop is at least 1x1, lies inside the image, contains the box scaled to pixels,
    and is at most one pixel larger than that box on each side."""
    width, height, box = case
    image = _coordinate_image(width, height)

    cropped = crop(image, box)

    left, top = _origin(cropped)
    right, bottom = left + cropped.width, top + cropped.height
    assert cropped.width >= 1 and cropped.height >= 1
    assert not _has_padding(cropped)
    assert 0 <= left and right <= width
    assert 0 <= top and bottom <= height
    assert left <= box.x_min * width and right >= box.x_max * width
    assert top <= box.y_min * height and bottom >= box.y_max * height
    assert cropped.width <= max(1, math.ceil((box.x_max - box.x_min) * width) + 2)
    assert cropped.height <= max(1, math.ceil((box.y_max - box.y_min) * height) + 2)


def test_crop_of_whole_image_box_is_the_whole_image():
    """A box covering the image returns an identical copy."""
    image = _coordinate_image(20, 10)
    cropped = crop(image, BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0))
    assert cropped.size == (20, 10)
    assert np.array_equal(np.asarray(cropped), np.asarray(image))


def test_crop_of_half_image():
    """The left half of a 20x10 image is the 10x10 block starting at the origin."""
    cropped = crop(_coordinate_image(20, 10), BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=1.0))
    assert cropped.size == (10, 10)
    assert _origin(cropped) == (0, 0)
    assert not _has_padding(cropped)


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(x_min=0.3, y_min=0.3, x_max=0.3, y_max=0.3),
        BoundingBox(x_min=1.0, y_min=1.0, x_max=1.0, y_max=1.0),
        BoundingBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0),
    ],
)
def test_crop_of_degenerate_box_is_one_pixel(box):
    """A zero-area box (also at the image border) still gives a 1x1 crop inside the image."""
    cropped = crop(_coordinate_image(20, 10), box)
    assert cropped.size == (1, 1)
    assert not _has_padding(cropped)
