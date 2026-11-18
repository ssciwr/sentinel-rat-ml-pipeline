"""Behavioural tests for the model handling module (`ml_pipeline.models.model`).

Layout follows the plan's test section:
- torch models: real pickled `.pt` files of tiny modules (tests/tiny_models.py)
- YOLO: the pretrained default models, downloaded once into the pytest cache
- SpeciesNet: an SDK stub (CI must not download the ~500 MB Kaggle model), plus an opt-in
  real-model test (`RUN_SPECIESNET_INTEGRATION=1`)

Models take one in-memory RGB PIL image per `predict` call; loading files is the service's job.
"""

import os
import socket
import sys
from pathlib import Path

import httpx2
import numpy as np
import pytest
import torch
import yaml
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageOps
from torchvision.transforms import v2

from ml_pipeline.models.model import (
    SpeciesnetClassifier,
    SpeciesnetDetector,
    TorchClassifier,
    TorchDetector,
    WeightsDownloadError,
    YoloClassifier,
    YoloDetector,
    build_model,
    build_transform,
    load_model,
    model_class_of,
    read_config,
    resolve_weights,
)
from ml_pipeline.schemas.analysis import BoundingBox, Classification, Detection, ModelInfo, Taxon
from tests import tiny_models

PROPERTY_SETTINGS = settings(deadline=None, max_examples=25)

INFO = {"name": "tiny-model", "version": "0", "description": "test model"}

TAXA = [
    {
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class_name": "Mammalia",
        "order": "Rodentia",
        "family": "Muridae",
        "genus": "Rattus",
        "species": "norvegicus",
        "common_name": "brown rat",
    },
    {
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class_name": "Mammalia",
        "order": "Rodentia",
        "family": "Muridae",
        "genus": "Rattus",
        "species": "rattus",
        "common_name": "black rat",
    },
    {
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class_name": "Mammalia",
        "order": "Rodentia",
        "family": "Muridae",
        "genus": "Mus",
        "species": "musculus",
        "common_name": "house mouse",
    },
]

DETECTOR_LABELS = ["animal", "human", "vehicle"]
DETECTOR_ROWS = [
    [0.1, 0.2, 0.4, 0.5, 0.6, 0.0],
    [0.2, 0.1, 0.9, 0.8, 0.95, 1.0],
    [0.3, 0.3, 0.6, 0.6, 0.05, 2.0],
    [-1.0, -1.0, -1.0, -1.0, 0.0, -5.0],  # padding: score 0, arbitrary (even invalid) box and class
]

SPECIESNET_V403B = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
SPECIESNET_V403A = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
# smallest float32 >= 0.01 (SpeciesNet's score floor); 0.01 itself is not representable in float32
SPECIESNET_FLOOR_F32 = float(np.nextafter(np.float32(0.01), np.float32(1.0)))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Session-wide directory for pickled models."""
    return tmp_path_factory.mktemp("model_data")


@pytest.fixture(scope="session")
def colour_images() -> list[Image.Image]:
    """Six 32x32 RGB images of distinct uniform colours, so model outputs differ per image."""
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (40, 40, 40)]
    return [Image.new("RGB", (32, 32), colour) for colour in colours]


def _save_module(module: object, path: Path) -> Path:
    """Pickle a whole module (or any object) with torch.save, the format the pipeline loads."""
    torch.save(module, path)
    return path


def _seeded(seed: int, build):
    """Build a module under a fixed seed without changing the global RNG state of other tests."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return build()


def _open_rgb(path: Path) -> Image.Image:
    """Open an image file as an EXIF-transposed RGB image (the service's job in production).

    Same steps as `ml_pipeline.services.images.load_image`, kept here on purpose so the model
    tests do not depend on the service layer.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@pytest.fixture(scope="session")
def logits_classifier_pt(data_dir) -> Path:
    """Tiny classifier with dropout that outputs logits for the three TAXA."""
    module = _seeded(0, lambda: tiny_models.TinyClassifier(len(TAXA)))
    return _save_module(module, data_dir / "logits_classifier.pt")


@pytest.fixture(scope="session")
def probability_classifier_pt(data_dir) -> Path:
    """Tiny classifier that outputs probabilities for the three TAXA."""
    module = _seeded(1, lambda: tiny_models.ProbabilityClassifier(len(TAXA)))
    return _save_module(module, data_dir / "prob_classifier.pt")


@pytest.fixture(scope="session")
def fixed_detector_pt(data_dir) -> Path:
    """Tiny detector returning DETECTOR_ROWS (incl. a padding row) for every image."""
    return _save_module(tiny_models.FixedRowsDetector(DETECTOR_ROWS), data_dir / "fixed_detector.pt")


@pytest.fixture(scope="session")
def colour_detector_pt(data_dir) -> Path:
    """Tiny detector with one colour-dependent box per image."""
    return _save_module(tiny_models.ColourDetector(), data_dir / "colour_detector.pt")


def make_torch_classifier(model_path: Path, **overrides) -> TorchClassifier:
    """Build a TorchClassifier with test defaults, overridable per test."""
    kwargs = {
        "info": INFO,
        "model_path": model_path,
        "preprocessing": {},
        "conf_limit": 0.0,
        "classes": TAXA,
        "apply_softmax": True,
    }
    kwargs.update(overrides)
    return TorchClassifier(**kwargs)


def make_torch_detector(model_path: Path, **overrides) -> TorchDetector:
    """Build a TorchDetector with test defaults, overridable per test."""
    kwargs = {
        "info": INFO,
        "model_path": model_path,
        "preprocessing": {},
        "conf_limit": 0.0,
        "labels": DETECTOR_LABELS,
    }
    kwargs.update(overrides)
    return TorchDetector(**kwargs)


def placeholder_taxa(count: int) -> list[dict]:
    """Minimal taxonomy entries (genus + species), one per class index."""
    return [{"genus": f"Genus{i}", "species": f"species{i}"} for i in range(count)]


def assert_classifier_payload(
    probs: torch.Tensor,
    classifications: list[Classification],
    taxa: list[dict],
    conf_limit: float,
    sum_atol: float,
) -> None:
    """Shared classifier invariants for one image: a CPU probability vector over all classes
    summing to 1, and classifications sorted descending, all >= conf_limit, complete w.r.t.
    conf_limit, each carrying the taxonomy and probability of its output index.
    (Species names are unique within every taxonomy used here.)"""
    assert probs.shape == (len(taxa),)
    assert probs.device.type == "cpu"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=sum_atol)
    confidences = [c.confidence for c in classifications]
    assert confidences == sorted(confidences, reverse=True)
    assert all(c >= conf_limit for c in confidences)
    assert len(classifications) == int((probs >= conf_limit).sum())
    species = [taxon["species"] for taxon in taxa]
    for classification in classifications:
        assert isinstance(classification, Classification)
        index = species.index(classification.species)
        assert classification.confidence == pytest.approx(float(probs[index]))
        assert classification.model_dump(exclude={"confidence"}) == Taxon(**taxa[index]).model_dump()


def assert_detector_payload(scores: torch.Tensor, detections: list[Detection], conf_limit: float) -> None:
    """Shared detector invariants for one image: CPU scores aligned with unclassified
    Detections, sorted descending, all >= conf_limit, boxes relative with min <= max."""
    assert scores.device.type == "cpu"
    assert scores.ndim == 1
    assert len(scores) == len(detections)
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)
    for score, detection in zip(scores.tolist(), detections):
        assert isinstance(detection, Detection)
        assert detection.confidence == pytest.approx(score)
        assert detection.confidence >= conf_limit
        assert detection.classifications == []
        assert isinstance(detection.bbox, BoundingBox)
        assert 0.0 <= detection.bbox.x_min <= detection.bbox.x_max <= 1.0
        assert 0.0 <= detection.bbox.y_min <= detection.bbox.y_max <= 1.0


def _box_coords(detections: list[Detection]) -> list[float]:
    """Corners of all boxes as one flat list (pytest.approx does not compare nested tuples)."""
    return [c for d in detections for c in (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max)]


# ------------------------------------------------------------------- torch classifier


def _raw_output(model_path: Path, image: Image.Image, *extra_transforms: v2.Transform) -> torch.Tensor:
    """The eval-mode module's output row for one image, fed by hand (float [0, 1] tensor plus
    `extra_transforms`): the reference the pipeline's predictions are compared with."""
    raw = torch.load(model_path, weights_only=False).eval()
    to_input = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), *extra_transforms])
    with torch.inference_mode():
        return raw(to_input(image).unsqueeze(0))[0]


def test_torch_classifier_runs_in_eval_mode(logits_classifier_pt, colour_images):
    """Dropout is inactive: predictions equal the eval-mode module's softmax output."""
    model = make_torch_classifier(logits_classifier_pt)
    for image in colour_images:
        probs, _ = model.predict(image)
        expected = torch.softmax(_raw_output(logits_classifier_pt, image), dim=0)
        assert torch.allclose(probs, expected, atol=1e-6)


@PROPERTY_SETTINGS
@given(conf_limit=st.floats(min_value=0.0, max_value=1.0, width=32))
def test_torch_classifier_payload_invariants(logits_classifier_pt, colour_images, conf_limit):
    """Payload: sorted descending, all >= conf_limit, taxonomy matches the output index,
    complete w.r.t. conf_limit; softmax-ed probability vectors sum to 1."""
    model = make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit)
    for image in colour_images:
        probs, classifications = model.predict(image)
        assert_classifier_payload(probs, classifications, TAXA, conf_limit, sum_atol=1e-5)


def test_torch_classifier_accepts_probabilities_without_softmax(probability_classifier_pt, colour_images):
    """With apply_softmax false the module's probabilities are used as they are."""
    model = make_torch_classifier(probability_classifier_pt, apply_softmax=False)
    for image in colour_images:
        probs, _ = model.predict(image)
        assert torch.allclose(probs, _raw_output(probability_classifier_pt, image), atol=1e-6)


def test_torch_preprocessing_runs_after_tensor_conversion(probability_classifier_pt, colour_images):
    """Configured transforms act on the float [0, 1] tensor: a `normalize` config equals
    feeding the raw module a hand-normalised input."""
    normalise = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    model = make_torch_classifier(
        probability_classifier_pt, apply_softmax=False, preprocessing={"normalize": normalise}
    )
    for image in colour_images:
        probs, _ = model.predict(image)
        expected = _raw_output(probability_classifier_pt, image, v2.Normalize(**normalise))
        assert torch.allclose(probs, expected, atol=1e-6)


def test_torch_classifier_takes_images_of_any_size(logits_classifier_pt):
    """One image per call: images of different sizes need no resize in the preprocessing."""
    model = make_torch_classifier(logits_classifier_pt)
    for size in [(32, 32), (48, 40), (7, 13)]:
        probs, _ = model.predict(Image.new("RGB", size, (10, 200, 30)))
        assert probs.shape == (len(TAXA),)


def test_torch_unimportable_model_class_raises_import_error(tmp_path):
    """A pickled module whose defining package is not installed gives an ImportError that says so."""
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "vanishing_models.py").write_text(
        "import torch\n\nclass Vanishing(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(package_dir))
        import vanishing_models

        path = _save_module(vanishing_models.Vanishing(), tmp_path / "vanishing.pt")
    # the package directory is off sys.path again; forget the imported module for good
    sys.modules.pop("vanishing_models", None)
    with pytest.raises(ImportError, match="vanishing_models"):
        make_torch_classifier(path)


def test_torch_pickled_non_module_raises(data_dir):
    """A .pt file that does not contain an nn.Module is rejected at construction."""
    path = _save_module({"weights": torch.zeros(1)}, data_dir / "not_a_module.pt")
    with pytest.raises(TypeError):
        make_torch_classifier(path)


def test_torch_preprocessing_must_produce_tensors(logits_classifier_pt, colour_images):
    """Torch models need tensors; a config that turns them back into PIL images is a TypeError."""
    model = make_torch_classifier(logits_classifier_pt, preprocessing={"to_pil_image": {}})
    with pytest.raises(TypeError):
        model.predict(colour_images[0])


@pytest.mark.parametrize(
    "content", [b"not a pickle at all", b"", b'{"weights": [1, 2, 3]}'], ids=["garbage", "empty", "json"]
)
def test_weights_that_are_no_torch_pickle_raise_value_error(tmp_path, content):
    """A weights file that torch.load cannot unpickle (safetensors, JSON, a truncated download) is
    an invalid config: ValueError naming the file."""
    path = tmp_path / "weights.pt"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="weights.pt"):
        make_torch_classifier(path)


def test_missing_local_weights_stay_a_file_not_found_error(tmp_path):
    """A missing local weights file is a config error (FileNotFoundError), not a download error."""
    with pytest.raises(FileNotFoundError):
        make_torch_classifier(tmp_path / "missing.pt")


# --------------------------------------------------------------------- torch detector


def test_torch_detector_payload_labels_and_padding(fixed_detector_pt, colour_images):
    """Detections are labelled from `labels` by class index, aligned with sorted scores;
    zero-score padding rows are dropped even at conf_limit 0 and never raise, although
    their boxes and class indices are invalid."""
    model = make_torch_detector(fixed_detector_pt, conf_limit=0.0)
    scores, detections = model.predict(colour_images[0])

    assert_detector_payload(scores, detections, conf_limit=0.0)
    assert scores.tolist() == pytest.approx([0.95, 0.6, 0.05])
    assert [d.detected_class for d in detections] == ["human", "animal", "vehicle"]
    assert _box_coords(detections) == pytest.approx([0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.4, 0.5, 0.3, 0.3, 0.6, 0.6])


@PROPERTY_SETTINGS
@given(conf_limit=st.floats(min_value=0.0, max_value=1.0, width=32))
def test_torch_detector_conf_limit(fixed_detector_pt, colour_images, conf_limit):
    """Only real rows with score >= conf_limit are kept, sorted descending, scores aligned."""
    model = make_torch_detector(fixed_detector_pt, conf_limit=conf_limit)
    scores, detections = model.predict(colour_images[0])
    scores32 = torch.tensor(DETECTOR_ROWS)[:, 4]  # float32, like the pipeline's comparison
    kept = scores32[(scores32 > 0) & (scores32 >= conf_limit)]
    assert scores.tolist() == pytest.approx(sorted(kept.tolist(), reverse=True))
    assert_detector_payload(scores, detections, conf_limit)


def test_torch_detector_boxes_follow_the_image(colour_detector_pt):
    """The detector's rows depend on the image it is given: red and blue images get
    different, correctly converted boxes and scores."""
    model = make_torch_detector(colour_detector_pt)
    red_scores, red = model.predict(Image.new("RGB", (32, 32), (255, 0, 0)))
    blue_scores, blue = model.predict(Image.new("RGB", (32, 32), (0, 0, 255)))

    assert red_scores.tolist() == pytest.approx([0.95])
    assert _box_coords(red) == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert blue_scores.tolist() == pytest.approx([0.05])
    assert _box_coords(blue) == pytest.approx([0.1, 0.1, 0.2, 0.9])
    assert [d.detected_class for d in red + blue] == ["animal", "animal"]


def test_torch_detector_checks_only_kept_rows(data_dir, colour_images):
    """A real row below conf_limit is dropped without validating its box and class index."""
    rows = [[0.1, 0.1, 0.5, 0.5, 0.9, 0.0], [2.0, -1.0, 0.5, 0.5, 0.3, 99.0]]
    path = _save_module(tiny_models.FixedRowsDetector(rows), data_dir / "below_limit.pt")
    scores, detections = make_torch_detector(path, conf_limit=0.5).predict(colour_images[0])
    assert scores.tolist() == pytest.approx([0.9])
    assert [d.detected_class for d in detections] == ["animal"]


def test_torch_detector_number_of_rows_may_vary_between_calls(data_dir, colour_images):
    """K may differ per call; each call returns its own rows."""
    row_sets = [
        [[0.1, 0.1, 0.2, 0.2, 0.9, 0.0]],
        [[0.1, 0.1, 0.2, 0.2, 0.9, 0.0], [0.3, 0.3, 0.4, 0.4, 0.8, 1.0]],
        [[0.1, 0.1, 0.2, 0.2, 0.9, 0.0], [0.3, 0.3, 0.4, 0.4, 0.8, 1.0], [0.5, 0.5, 0.6, 0.6, 0.7, 2.0]],
    ]
    path = _save_module(tiny_models.VaryingRowsDetector(row_sets), data_dir / "varying.pt")
    model = make_torch_detector(path)
    counts = [len(model.predict(image)[1]) for image in colour_images[:3]]
    assert counts == [1, 2, 3]


def test_label_map_renames_mapped_labels_only(fixed_detector_pt, colour_images):
    """Mapped labels are renamed; unmapped labels are returned verbatim."""
    model = make_torch_detector(fixed_detector_pt, label_map={"animal": "rodent", "vehicle": "car"})
    _, detections = model.predict(colour_images[0])
    assert [d.detected_class for d in detections] == ["human", "rodent", "car"]


def test_label_map_with_unknown_label_raises(fixed_detector_pt):
    """A label_map key the model cannot emit (e.g. a typo) is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="anmal"):
        make_torch_detector(fixed_detector_pt, label_map={"anmal": "rodent"})


@pytest.mark.parametrize(
    ("label_map", "error"),
    [
        (["animal"], TypeError),
        ({"animal": 1}, TypeError),
        ({1: "animal"}, TypeError),
        ({"animal": ""}, ValueError),
        ({"": "animal"}, ValueError),
    ],
)
def test_invalid_label_map_fails_before_download(hub_stub, label_map, error):
    """label_map must map non-empty strings to non-empty strings; checked before any weights."""
    with pytest.raises(error):
        TorchDetector(
            info=INFO,
            hf_repo="org/repo",
            hf_filename="det.pt",
            preprocessing={},
            conf_limit=0.1,
            labels=DETECTOR_LABELS,
            label_map=label_map,
        )
    assert hub_stub == []


@pytest.mark.parametrize(
    ("labels", "error"),
    [
        ([], ValueError),
        (["animal", "animal"], ValueError),
        (["animal", ""], ValueError),
        (["animal", 1], TypeError),
        ("animal", TypeError),
    ],
)
def test_invalid_torch_detector_labels_fail_before_download(hub_stub, labels, error):
    """labels must be a non-empty list of unique, non-empty strings; checked before any weights."""
    with pytest.raises(error):
        TorchDetector(
            info=INFO,
            hf_repo="org/repo",
            hf_filename="det.pt",
            preprocessing={},
            conf_limit=0.1,
            labels=labels,
        )
    assert hub_stub == []


def test_torch_detector_requires_labels(fixed_detector_pt):
    """TorchDetector has no default label set."""
    with pytest.raises(TypeError):
        TorchDetector(info=INFO, model_path=fixed_detector_pt, preprocessing={}, conf_limit=0.1)


# ------------------------------------------------------------- torch contract checks


@pytest.mark.parametrize(
    ("mode", "apply_softmax"),
    [
        ("tuple", True),
        ("rank3", True),
        ("batch", True),
        ("width", True),
        ("not_normalised", False),
        ("out_of_range", False),
        ("nan", False),
        ("nan", True),
    ],
)
def test_torch_classifier_contract_violations_raise(data_dir, colour_images, mode, apply_softmax):
    """Every violation of the classifier output contract raises a ValueError."""
    path = _save_module(tiny_models.ContractBreaker(mode, width=len(TAXA)), data_dir / f"cls_{mode}.pt")
    model = make_torch_classifier(path, apply_softmax=apply_softmax)
    with pytest.raises(ValueError):
        model.predict(colour_images[0])


@pytest.mark.parametrize(
    "mode",
    [
        "det_tuple",
        "det_rank2",
        "det_five_columns",
        "det_batch",
        "det_score_range",
        "det_score_nan",
        "det_score_negative",
        "det_box_range",
        "det_box_inverted",
        "det_box_inverted_y",
        "det_class_fraction",
        "det_class_negative",
        "det_class_too_large",
    ],
)
def test_torch_detector_contract_violations_raise(data_dir, colour_images, mode):
    """Every violation of the detector output contract raises a ValueError."""
    path = _save_module(tiny_models.ContractBreaker(mode), data_dir / f"det_{mode}.pt")
    model = make_torch_detector(path, conf_limit=0.25)
    with pytest.raises(ValueError):
        model.predict(colour_images[0])


# ------------------------------------------------------------------ predict input


@pytest.mark.parametrize(
    "not_an_image",
    [
        [Image.new("RGB", (8, 8))],
        "image.jpg",
        Path("image.jpg"),
        np.zeros((8, 8, 3), dtype=np.uint8),
        torch.zeros(3, 8, 8),
    ],
)
def test_predict_takes_one_pil_image(logits_classifier_pt, fixed_detector_pt, not_an_image):
    """predict takes exactly one PIL image: lists, paths, arrays and tensors are TypeErrors."""
    for model in [make_torch_classifier(logits_classifier_pt), make_torch_detector(fixed_detector_pt)]:
        with pytest.raises(TypeError):
            model.predict(not_an_image)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_predict_rejects_non_rgb_images(logits_classifier_pt, fixed_detector_pt, mode):
    """Models expect RGB input; converting is the caller's job."""
    for model in [make_torch_classifier(logits_classifier_pt), make_torch_detector(fixed_detector_pt)]:
        with pytest.raises(ValueError):
            model.predict(Image.new(mode, (8, 8)))


# ------------------------------------------------------------------- build_transform


def _random_image(seed: int) -> Image.Image:
    """A reproducible random 24x20 RGB image."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8))


@pytest.mark.parametrize("preprocessing", [{}, None])
def test_build_transform_empty_is_identity(preprocessing):
    """An empty node (`{}` or YAML null) leaves the image unchanged."""
    image = _random_image(0)
    result = build_transform(preprocessing)(image)
    assert np.array_equal(np.asarray(result), np.asarray(image))


DETERMINISTIC_TRANSFORMS = [
    ("resize", {"size": [16, 16], "interpolation": "bilinear", "antialias": True}, v2.Resize),
    ("center_crop", {"size": [10, 10]}, v2.CenterCrop),
    ("grayscale", {"num_output_channels": 3}, v2.Grayscale),
    ("random_horizontal_flip", {"p": 1.0}, v2.RandomHorizontalFlip),
    ("pad", {"padding": 2}, v2.Pad),
]


@PROPERTY_SETTINGS
@given(
    order=st.permutations(DETERMINISTIC_TRANSFORMS),
    count=st.integers(min_value=0, max_value=len(DETERMINISTIC_TRANSFORMS)),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_build_transform_matches_hand_composed_transforms(order, count, seed):
    """Snake_case keys build the same pipeline, in mapping order, as composing the v2
    transforms by hand."""
    chosen = order[:count]
    preprocessing = {name: kwargs for name, kwargs, _ in chosen}
    image = _random_image(seed)

    result = build_transform(preprocessing)(image)

    expected = image
    for _, kwargs, transform_class in chosen:
        expected = transform_class(**kwargs)(expected)
    assert np.array_equal(np.asarray(result), np.asarray(expected))


def test_build_transform_name_matching_ignores_case_and_underscores():
    """`pil_to_tensor` resolves to PILToTensor, CamelCase keys work as well."""
    image = _random_image(1)
    assert isinstance(build_transform({"pil_to_tensor": {}})(image), torch.Tensor)
    flipped = build_transform({"RandomHorizontalFlip": {"p": 1.0}})(image)
    assert np.array_equal(np.asarray(flipped), np.asarray(image)[:, ::-1])


@pytest.mark.parametrize("name", ["no_such_transform", "functional", "query_size", "transform"])
def test_build_transform_rejects_unknown_and_non_transform_names(name):
    """Only concrete v2.Transform subclasses resolve; anything else (including the abstract
    base `Transform`) is a ValueError."""
    with pytest.raises(ValueError):
        build_transform({name: {}})


@pytest.mark.parametrize("kwargs", [16, [16, 16], "bilinear"])
def test_build_transform_rejects_non_mapping_kwargs(kwargs):
    """Transform kwargs must be a mapping."""
    with pytest.raises(TypeError):
        build_transform({"resize": kwargs})


@pytest.mark.parametrize("name", [1, None, ("resize",)])
def test_build_transform_rejects_non_string_names(name):
    """Transform names must be strings (e.g. YAML `1: {}` is an error)."""
    with pytest.raises(TypeError):
        build_transform({name: {}})


@pytest.mark.parametrize("preprocessing", [["resize"], "resize", 3])
def test_build_transform_rejects_non_mapping_node(preprocessing):
    """The preprocessing node itself must be a mapping (or empty)."""
    with pytest.raises(TypeError):
        build_transform(preprocessing)


# --------------------------------------------------------- constructor validation


def test_model_info_is_kept(logits_classifier_pt, fixed_detector_pt):
    """The configured info is available as a ModelInfo, for the API response."""
    assert make_torch_classifier(logits_classifier_pt).info == ModelInfo(**INFO)
    assert make_torch_detector(fixed_detector_pt).info == ModelInfo(**INFO)


def test_missing_info_raises(logits_classifier_pt):
    """info is required: the response must name the model that ran."""
    with pytest.raises(TypeError):
        TorchClassifier(
            model_path=logits_classifier_pt, preprocessing={}, conf_limit=0.1, classes=TAXA, apply_softmax=True
        )


@pytest.mark.parametrize("info", [{}, {"version": "1"}, "tiny-model", {"name": "x", "verison": "1"}])
def test_invalid_info_fails_before_download(hub_stub, info):
    """info must be a valid ModelInfo (a mapping with a name, no misspelled keys); checked
    before any weights."""
    with pytest.raises(ValueError):
        TorchClassifier(
            info=info,
            hf_repo="org/repo",
            hf_filename="cls.pt",
            preprocessing={},
            conf_limit=0.1,
            classes=TAXA,
            apply_softmax=True,
        )
    assert hub_stub == []


def test_missing_preprocessing_raises(logits_classifier_pt):
    """The preprocessing node is required."""
    with pytest.raises(TypeError):
        TorchClassifier(info=INFO, model_path=logits_classifier_pt, conf_limit=0.1, classes=TAXA, apply_softmax=True)


def test_torch_classifier_requires_apply_softmax(logits_classifier_pt):
    """apply_softmax has no default: logits vs probabilities must be stated."""
    with pytest.raises(TypeError):
        TorchClassifier(info=INFO, model_path=logits_classifier_pt, preprocessing={}, conf_limit=0.1, classes=TAXA)


def test_torch_classifier_requires_classes(logits_classifier_pt):
    """classes has no default: every output index needs its taxonomy."""
    with pytest.raises(TypeError):
        TorchClassifier(
            info=INFO, model_path=logits_classifier_pt, preprocessing={}, conf_limit=0.1, apply_softmax=True
        )


@pytest.mark.parametrize("classes", ["Rattus", {"genus": "Rattus"}, 3])
def test_non_list_classes_fail_before_download(hub_stub, classes):
    """classes must be a list of taxonomy entries (a single mapping or string is an error)."""
    with pytest.raises(TypeError):
        TorchClassifier(
            info=INFO,
            hf_repo="org/repo",
            hf_filename="cls.pt",
            preprocessing={},
            conf_limit=0.1,
            classes=classes,
            apply_softmax=True,
        )
    assert hub_stub == []


def test_yolo_classifier_requires_classes(tmp_path):
    """classes has no default for YOLO classifiers either (checked before any weights)."""
    with pytest.raises(TypeError):
        YoloClassifier(info=INFO, model_path=tmp_path / "missing.pt", preprocessing={}, conf_limit=0.1)


@pytest.mark.parametrize("conf_limit", [0, 1])
def test_integer_conf_limit_is_accepted(logits_classifier_pt, colour_images, conf_limit):
    """YAML `conf_limit: 0` / `1` are ints; they are valid limits (at 1, typically nothing is kept)."""
    probs, classifications = make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit).predict(
        colour_images[0]
    )
    assert len(classifications) == int((probs >= conf_limit).sum())


def test_batch_size_is_not_an_argument(logits_classifier_pt):
    """There is no batching: batch_size is an unknown argument."""
    with pytest.raises(TypeError):
        make_torch_classifier(logits_classifier_pt, batch_size=4)


def test_model_path_and_hf_repo_together_raise(logits_classifier_pt, hub_stub):
    """Weights come from exactly one source."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, hf_repo="org/repo", hf_filename="m.pt")
    assert hub_stub == []


@pytest.mark.parametrize("model_class", [TorchClassifier, TorchDetector, YoloClassifier, YoloDetector])
def test_neither_model_path_nor_hf_repo_raises(model_class):
    """Without a default (only SpeciesNet has one), a weights source is required."""
    kwargs = {"info": INFO, "preprocessing": {}, "conf_limit": 0.1}
    if model_class in (TorchClassifier, YoloClassifier):
        kwargs["classes"] = TAXA
    if model_class is TorchClassifier:
        kwargs["apply_softmax"] = True
    if model_class is TorchDetector:
        kwargs["labels"] = DETECTOR_LABELS
    with pytest.raises(ValueError):
        model_class(**kwargs)


def test_hf_repo_without_filename_raises(hub_stub):
    """A Hub download needs the file name."""
    with pytest.raises(ValueError):
        TorchClassifier(
            info=INFO, hf_repo="org/repo", preprocessing={}, conf_limit=0.1, classes=TAXA, apply_softmax=True
        )
    assert hub_stub == []


@pytest.mark.parametrize("stray", [{"hf_filename": "m.pt"}, {"hf_revision": "v1"}])
def test_hub_arguments_without_hf_repo_raise(logits_classifier_pt, hub_stub, stray):
    """hf_filename / hf_revision next to a local model_path are errors, not silently ignored."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, **stray)
    assert hub_stub == []


@pytest.mark.parametrize("conf_limit", [-0.1, 1.1, float("nan")])
def test_conf_limit_outside_unit_interval_raises(logits_classifier_pt, conf_limit):
    """conf_limit must lie in [0, 1]."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit)


@pytest.mark.parametrize("conf_limit", ["0.5", None, True])
def test_non_numeric_conf_limit_raises(logits_classifier_pt, conf_limit):
    """conf_limit must be a number (bool is rejected explicitly)."""
    with pytest.raises(TypeError):
        make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit)


@pytest.mark.parametrize("apply_softmax", ["yes", 1, None])
def test_non_bool_apply_softmax_raises(logits_classifier_pt, apply_softmax):
    """apply_softmax must be true or false."""
    with pytest.raises(TypeError):
        make_torch_classifier(logits_classifier_pt, apply_softmax=apply_softmax)


# ---------------------------------------------------------------------- config files


def _write_config(path: Path, config) -> Path:
    """Dump a config object to a YAML file."""
    path.write_text(yaml.safe_dump(config))
    return path


def _classifier_kwargs(model_path: Path) -> dict:
    """Complete TorchClassifier keyword arguments (a config file adds `type`)."""
    return {
        "info": INFO,
        "model_path": str(model_path),
        "preprocessing": {"resize": {"size": [16, 16], "antialias": True}},
        "conf_limit": 0.1,
        "classes": TAXA,
        "apply_softmax": True,
    }


def test_read_config_returns_the_mapping(logits_classifier_pt, tmp_path):
    """read_config parses a YAML config into its mapping, `type` included."""
    config = {"type": "TorchClassifier"} | _classifier_kwargs(logits_classifier_pt)
    assert read_config(_write_config(tmp_path / "cls.yaml", config)) == config


@pytest.mark.parametrize("content", [["a", "b"], "just a string", 42])
def test_read_config_non_mapping_top_level_raises(tmp_path, content):
    """A config file must contain a mapping at the top level."""
    with pytest.raises(TypeError):
        read_config(_write_config(tmp_path / "cls.yaml", content))


def test_build_model_from_a_mapping(logits_classifier_pt, colour_images):
    """build_model builds the class named in `type` from an in-memory config, without
    changing the caller's mapping."""
    config = {"type": "TorchClassifier"} | _classifier_kwargs(logits_classifier_pt)
    original = dict(config)

    model = build_model(config)

    assert type(model) is TorchClassifier
    assert config == original
    probs, _ = model.predict(colour_images[0])
    assert probs.shape == (len(TAXA),)


@pytest.mark.parametrize("config", [["type", "TorchClassifier"], "TorchClassifier", None])
def test_build_model_non_mapping_raises(config):
    """An in-memory config must be a mapping."""
    with pytest.raises(TypeError):
        build_model(config)


def test_weights_describe_a_local_model_path(logits_classifier_pt):
    """A model built from a local file names that file as its weights."""
    assert make_torch_classifier(logits_classifier_pt).weights == str(logits_classifier_pt)


@pytest.mark.parametrize(
    ("model_type", "model_class"),
    [("TorchDetector", TorchDetector), ("SpeciesnetClassifier", SpeciesnetClassifier), ("YoloDetector", YoloDetector)],
)
def test_model_class_of_names_the_class_without_building(model_type, model_class):
    """The class a config names is known without loading anything (the other keys may be invalid)."""
    assert model_class_of({"type": model_type, "model_path": "/nonexistent.pt"}) is model_class


@pytest.mark.parametrize(("config", "error"), [({}, ValueError), ({"type": "ResNet"}, ValueError), ([], TypeError)])
def test_model_class_of_rejects_unknown_or_missing_types(config, error):
    """A config without a known `type` (or no mapping at all) names no class."""
    with pytest.raises(error):
        model_class_of(config)


def test_load_model_matches_keyword_construction(logits_classifier_pt, colour_images, tmp_path):
    """A model loaded from a config file is of the class named in `type` and predicts exactly
    like one built from the same kwargs."""
    kwargs = _classifier_kwargs(logits_classifier_pt)
    from_file = load_model(_write_config(tmp_path / "cls.yaml", {"type": "TorchClassifier"} | kwargs))
    from_kwargs = TorchClassifier(**kwargs)

    assert type(from_file) is TorchClassifier
    assert from_file.info == from_kwargs.info
    for image in colour_images:
        probs_file, payload_file = from_file.predict(image)
        probs_kwargs, payload_kwargs = from_kwargs.predict(image)
        assert torch.equal(probs_file, probs_kwargs)
        assert payload_file == payload_kwargs


def test_load_model_builds_a_torch_detector(fixed_detector_pt, colour_images, tmp_path):
    """`type: TorchDetector` builds a detector from the remaining keys."""
    config = {
        "type": "TorchDetector",
        "info": INFO,
        "model_path": str(fixed_detector_pt),
        "preprocessing": {},
        "conf_limit": 0.5,
        "labels": DETECTOR_LABELS,
        "label_map": {"human": "person"},
    }
    model = load_model(_write_config(tmp_path / "det.yaml", config))
    assert type(model) is TorchDetector
    _, detections = model.predict(colour_images[0])
    assert [d.detected_class for d in detections] == ["person", "animal"]


def test_load_model_builds_speciesnet(speciesnet_stub, tmp_path):
    """`type: SpeciesnetDetector` builds SpeciesNet, which needs no weights source."""
    config = {"type": "SpeciesnetDetector", "info": INFO, "preprocessing": {}, "conf_limit": 0.1}
    model = load_model(_write_config(tmp_path / "speciesnet.yaml", config))
    assert type(model) is SpeciesnetDetector
    assert speciesnet_stub.instances[-1].model_name == SPECIESNET_V403B


def test_load_model_with_null_preprocessing_node(logits_classifier_pt, colour_images, tmp_path):
    """A bare `preprocessing:` (YAML null) counts as an empty node."""
    path = tmp_path / "cls.yaml"
    path.write_text(
        f"type: TorchClassifier\nmodel_path: {logits_classifier_pt}\npreprocessing:\nconf_limit: 0.1\n"
        "apply_softmax: true\n" + yaml.safe_dump({"info": INFO, "classes": TAXA})
    )
    probs, _ = load_model(path).predict(colour_images[0])
    assert probs.shape == (len(TAXA),)


@pytest.mark.parametrize("model_type", [None, "ResNet", "torchclassifier", "ModelBase", 3])
def test_load_model_unknown_or_missing_type_raises(logits_classifier_pt, tmp_path, model_type):
    """`type` must name one of the model classes exactly."""
    config = _classifier_kwargs(logits_classifier_pt)
    if model_type is not None:
        config["type"] = model_type
    with pytest.raises(ValueError):
        load_model(_write_config(tmp_path / "cls.yaml", config))


@pytest.mark.parametrize("removed", ["preprocessing", "info"])
def test_load_model_missing_required_key_raises(logits_classifier_pt, tmp_path, removed):
    """Required keys are required in config files, too."""
    config = {"type": "TorchClassifier"} | _classifier_kwargs(logits_classifier_pt)
    del config[removed]
    with pytest.raises(TypeError):
        load_model(_write_config(tmp_path / "cls.yaml", config))


@pytest.mark.parametrize("extra", [{"learning_rate": 0.1}, {"batch_size": 4}])
def test_load_model_unknown_key_raises(logits_classifier_pt, tmp_path, extra):
    """Unknown config keys are errors, not silently ignored."""
    config = {"type": "TorchClassifier"} | _classifier_kwargs(logits_classifier_pt) | extra
    with pytest.raises(TypeError):
        load_model(_write_config(tmp_path / "cls.yaml", config))


# ------------------------------------------------------------------ Hugging Face Hub


@pytest.fixture
def hub_stub(monkeypatch, logits_classifier_pt):
    """Replace hf_hub_download; record calls and return the local tiny classifier."""
    calls = []

    def fake_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        return str(logits_classifier_pt)

    monkeypatch.setattr("ml_pipeline.models.model.hf_hub_download", fake_hf_hub_download)
    return calls


@pytest.mark.parametrize(
    "error",
    [
        LocalEntryNotFoundError("not cached and the Hub is unreachable"),
        OfflineModeIsEnabled("HF_HUB_OFFLINE is set"),
        OSError("connection reset"),
        httpx2.ReadError("connection closed while reading"),
        httpx2.ProxyError("proxy refused the connection"),
    ],
    ids=["not_cached", "offline", "network", "read_error", "proxy_error"],
)
def test_failing_hub_download_raises_weights_download_error(monkeypatch, error):
    """Download failures (huggingface_hub's OSErrors and httpx2's network errors) become
    WeightsDownloadError, which is not an OSError, so they cannot be mistaken for config errors
    such as a missing local file."""

    def failing_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        raise error

    monkeypatch.setattr("ml_pipeline.models.model.hf_hub_download", failing_hf_hub_download)
    with pytest.raises(WeightsDownloadError) as excinfo:
        resolve_weights(None, "org/rodent-models", "cls.pt", None)
    assert excinfo.value.__cause__ is error
    assert not issubclass(WeightsDownloadError, OSError)


@pytest.mark.parametrize(
    ("revision", "weights"),
    [(None, "hf:org/rodent-models/weights/classifier.pt"), ("v1.2", "hf:org/rodent-models/weights/classifier.pt@v1.2")],
)
def test_hf_download_is_used_and_loaded(hub_stub, colour_images, revision, weights):
    """HF args are forwarded to hf_hub_download, the downloaded file is loaded, and the model
    names repo, file and (if given) revision as its weights."""
    model = TorchClassifier(
        info=INFO,
        hf_repo="org/rodent-models",
        hf_filename="weights/classifier.pt",
        hf_revision=revision,
        preprocessing={},
        conf_limit=0.1,
        classes=TAXA,
        apply_softmax=True,
    )
    assert hub_stub == [{"repo_id": "org/rodent-models", "filename": "weights/classifier.pt", "revision": revision}]
    assert model.weights == weights
    probs, _ = model.predict(colour_images[0])
    assert probs.shape == (len(TAXA),)


@pytest.mark.parametrize(
    "bad_args",
    [
        {"preprocessing": {"no_such_transform": {}}},
        {"classes": [{"genus": ["Rattus"]}]},  # genus must be a string
        {"classes": [{"genus": "Rattus", "specis": "norvegicus"}]},  # misspelled key
        {"classes": []},  # a classifier needs at least one class
    ],
)
def test_invalid_config_fails_before_download(hub_stub, bad_args):
    """Config errors surface before any weights are resolved or downloaded."""
    kwargs = {
        "info": INFO,
        "hf_repo": "org/rodent-models",
        "hf_filename": "weights/classifier.pt",
        "preprocessing": {},
        "conf_limit": 0.1,
        "classes": TAXA,
        "apply_softmax": True,
    } | bad_args
    with pytest.raises(ValueError):
        TorchClassifier(**kwargs)
    assert hub_stub == []


# --------------------------------------------------------- returned tensors (all types)


def _assert_ordinary_tensor(tensor: torch.Tensor) -> None:
    """A returned tensor is a normal CPU tensor that callers may modify in place."""
    assert tensor.device.type == "cpu"
    assert not tensor.is_inference()
    tensor.mul_(1.0)  # raises for inference tensors


@pytest.mark.parametrize("apply_softmax", [True, False])
def test_torch_classifier_returns_ordinary_tensors(probability_classifier_pt, colour_images, apply_softmax):
    """No inference-mode tensors leak out of predict, whatever the output dtype path."""
    model = make_torch_classifier(probability_classifier_pt, apply_softmax=apply_softmax)
    _assert_ordinary_tensor(model.predict(colour_images[0])[0])


def test_torch_detector_returns_ordinary_tensors(fixed_detector_pt, colour_images):
    """No inference-mode tensors leak out of a detector's predict."""
    _assert_ordinary_tensor(make_torch_detector(fixed_detector_pt).predict(colour_images[0])[0])


# ---------------------------------------------------------------------------- YOLO


@pytest.fixture(scope="session")
def yolo_weights(pytestconfig):
    """Return a function giving the path of a pretrained YOLO model in the pytest cache.

    Ultralytics downloads known asset names to the given path on first use, so the
    weights are fetched once and reused across runs. Without network or cache this
    fails loudly instead of skipping, so YOLO coverage cannot silently disappear.
    """
    from ultralytics import YOLO

    cache = getattr(pytestconfig, "cache", None)
    if cache is None:
        pytest.fail("the YOLO tests keep their weights in the pytest cache; do not run with -p no:cacheprovider")
    weights_dir = cache.mkdir("yolo_weights")

    def weights(name: str) -> Path:
        path = weights_dir / name
        try:
            if not path.exists():
                YOLO(str(path))
        except Exception as error:  # any failure here means the fixture is unusable
            pytest.fail(f"pretrained YOLO weights not cached and download failed: {error}")
        return path

    return weights


@pytest.fixture(scope="session")
def ultralytics_images() -> dict[str, Path]:
    """Sample photographs bundled with ultralytics."""
    from ultralytics.utils import ASSETS

    return {"bus": ASSETS / "bus.jpg", "zidane": ASSETS / "zidane.jpg"}


def _direct_yolo_detections(weights: Path, image: Image.Image, conf: float) -> list[tuple[float, str]]:
    """(score, COCO name) of every box of a plain ultralytics call, sorted by descending score."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.predict(source=image, conf=conf, device="cpu", verbose=False)[0]
    boxes = result.boxes
    pairs = [(score, model.names[int(cls)]) for score, cls in zip(boxes.conf.tolist(), boxes.cls.tolist())]
    return sorted(pairs, key=lambda pair: pair[0], reverse=True)


def test_yolo_detector_returns_labelled_boxes_of_all_classes(yolo_weights, ultralytics_images):
    """The pretrained COCO detector: every box of every class is kept and labelled with its
    COCO name, with the same scores as a plain ultralytics call, sorted above conf_limit."""
    weights = yolo_weights("yolo26n.pt")
    model = YoloDetector(info=INFO, model_path=weights, preprocessing={}, conf_limit=0.25)
    image = _open_rgb(ultralytics_images["bus"])
    scores, detections = model.predict(image)

    expected = _direct_yolo_detections(weights, image, conf=0.25)
    assert scores.tolist() == pytest.approx([score for score, _ in expected], abs=1e-4)
    # compared as sorted (score, label) pairs, so exactly tied scores cannot flip the order
    returned = sorted((round(d.confidence, 4), d.detected_class) for d in detections)
    assert returned == sorted((round(score, 4), name) for score, name in expected)
    assert {"person", "bus"} <= {d.detected_class for d in detections}
    assert_detector_payload(scores, detections, conf_limit=0.25)
    _assert_ordinary_tensor(scores)


def test_yolo_label_map_renames_coco_labels(yolo_weights, ultralytics_images):
    """label_map renames COCO names: `bus` becomes `vehicle`, other labels stay."""
    model = YoloDetector(
        info=INFO,
        model_path=yolo_weights("yolo26n.pt"),
        preprocessing={},
        conf_limit=0.25,
        label_map={"bus": "vehicle"},
    )
    _, detections = model.predict(_open_rgb(ultralytics_images["bus"]))
    labels = {d.detected_class for d in detections}
    assert "vehicle" in labels
    assert "bus" not in labels
    assert "person" in labels


def test_yolo_label_map_with_unknown_label_raises(yolo_weights):
    """A label_map key that is no COCO name of the loaded model is rejected after loading."""
    with pytest.raises(ValueError, match="buss"):
        YoloDetector(
            info=INFO,
            model_path=yolo_weights("yolo26n.pt"),
            preprocessing={},
            conf_limit=0.25,
            label_map={"buss": "vehicle"},
        )


def test_yolo_applies_configured_preprocessing(yolo_weights, ultralytics_images):
    """Configured transforms run before YOLO: a horizontal flip config gives the same
    detections as running ultralytics on the mirrored image."""
    weights = yolo_weights("yolo26n.pt")
    model = YoloDetector(
        info=INFO, model_path=weights, preprocessing={"random_horizontal_flip": {"p": 1.0}}, conf_limit=0.25
    )
    image = _open_rgb(ultralytics_images["bus"])
    scores, _ = model.predict(image)
    expected = _direct_yolo_detections(weights, ImageOps.mirror(image), conf=0.25)
    assert scores.tolist() == pytest.approx([score for score, _ in expected], abs=1e-4)


def test_yolo_missing_non_asset_path_raises_without_network(tmp_path, monkeypatch):
    """A missing file that is not an ultralytics asset name fails immediately and offline."""
    attempts = []

    def record_connect(self, address):
        attempts.append(address)
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", record_connect)
    with pytest.raises(FileNotFoundError):
        YoloDetector(info=INFO, model_path=tmp_path / "rodnet.pt", preprocessing={}, conf_limit=0.25)
    assert attempts == []


def test_yolo_rejects_non_pil_preprocessing_output(yolo_weights, ultralytics_images):
    """YOLO preprocesses internally, so configured transforms must keep PIL images."""
    model = YoloDetector(
        info=INFO, model_path=yolo_weights("yolo26n.pt"), preprocessing={"pil_to_tensor": {}}, conf_limit=0.25
    )
    with pytest.raises(TypeError):
        model.predict(_open_rgb(ultralytics_images["bus"]))


def test_yolo_classifier_maps_probabilities_to_taxonomy(yolo_weights, ultralytics_images):
    """The pretrained ImageNet classifier: full probability vector plus sorted,
    conf_limit-filtered classifications carrying the configured taxonomy."""
    taxa = placeholder_taxa(1000)  # one entry per ImageNet class
    model = YoloClassifier(
        info=INFO, model_path=yolo_weights("yolo26n-cls.pt"), preprocessing={}, conf_limit=0.05, classes=taxa
    )
    probs, classifications = model.predict(_open_rgb(ultralytics_images["bus"]))

    assert_classifier_payload(probs, classifications, taxa, conf_limit=0.05, sum_atol=1e-4)
    _assert_ordinary_tensor(probs)


def test_load_model_builds_yolo_models(yolo_weights, tmp_path):
    """`type: YoloDetector` / `type: YoloClassifier` build the YOLO models."""
    detector_config = {
        "type": "YoloDetector",
        "info": INFO,
        "model_path": str(yolo_weights("yolo26n.pt")),
        "preprocessing": {},
        "conf_limit": 0.25,
    }
    classifier_config = {
        "type": "YoloClassifier",
        "info": INFO,
        "model_path": str(yolo_weights("yolo26n-cls.pt")),
        "preprocessing": {},
        "conf_limit": 0.05,
        "classes": placeholder_taxa(1000),
    }
    assert type(load_model(_write_config(tmp_path / "det.yaml", detector_config))) is YoloDetector
    assert type(load_model(_write_config(tmp_path / "cls.yaml", classifier_config))) is YoloClassifier


def test_yolo_weights_of_the_wrong_task_raise(yolo_weights):
    """Classifier weights for a detector (and vice versa) are rejected at construction."""
    with pytest.raises(ValueError, match="classify"):
        YoloDetector(info=INFO, model_path=yolo_weights("yolo26n-cls.pt"), preprocessing={}, conf_limit=0.25)
    with pytest.raises(ValueError, match="detect"):
        YoloClassifier(
            info=INFO,
            model_path=yolo_weights("yolo26n.pt"),
            preprocessing={},
            conf_limit=0.05,
            classes=placeholder_taxa(80),
        )


def test_yolo_classifier_taxonomy_count_mismatch_raises(yolo_weights):
    """The number of taxonomy entries must equal the model's number of classes."""
    with pytest.raises(ValueError):
        YoloClassifier(
            info=INFO,
            model_path=yolo_weights("yolo26n-cls.pt"),
            preprocessing={},
            conf_limit=0.05,
            classes=placeholder_taxa(999),
        )


# ------------------------------------------------------------------ SpeciesNet (stub)


def make_speciesnet_stub(detections: list[dict]):
    """Create a stand-in for `speciesnet.detector.SpeciesNetDetector`.

    It mimics the SDK surface used by the pipeline: construction from a model name,
    `device` / `model` attributes (initially not on CPU, like a GPU host), and
    `preprocess(PIL image)` / `predict(filepath, preprocessed)` returning SDK-shaped
    detections (relative [x_min, y_min, width, height] boxes, sorted by confidence).
    `predict` only accepts what `preprocess` returned, so skipping the SDK's own
    preprocessing fails. Every instance records the filepaths `predict` received.
    """
    instances = []

    class Preprocessed:
        """Marker for the output of the stub's `preprocess`."""

        def __init__(self, image):
            self.image = image

    class RecordingModule(torch.nn.Linear):
        """A tiny model that records whether it was moved (`.to()` / `.cpu()` both apply)."""

        moved = False

        def _apply(self, fn, recurse=True):
            self.moved = True
            return super()._apply(fn, recurse)

    class StubSpeciesNetDetector:
        """Stand-in for the SDK detector component."""

        def __init__(self, model_name, force_model_download=False):
            self.model_name = model_name
            self.device = "cuda"
            self.model = RecordingModule(2, 2)
            self.filepaths = []
            instances.append(self)

        def preprocess(self, img):
            """Accept only RGB PIL images, as the SDK does."""
            assert isinstance(img, Image.Image) and img.mode == "RGB"
            return Preprocessed(img)

        def predict(self, filepath, img):
            """Return the configured detections, sorted by confidence like the SDK."""
            assert isinstance(img, Preprocessed), "predict must get the output of preprocess"
            self.filepaths.append(filepath)
            ordered = sorted(detections, key=lambda d: d["conf"], reverse=True)
            return {"filepath": filepath, "detections": [dict(d) for d in ordered]}

    StubSpeciesNetDetector.instances = instances
    return StubSpeciesNetDetector


def _sdk_detection(label: str, conf: float, bbox: list[float]) -> dict:
    """One detection in the SpeciesNet SDK output format."""
    category = {"animal": "1", "human": "2", "vehicle": "3"}[label]
    return {"category": category, "label": label, "conf": conf, "bbox": bbox}


@st.composite
def sdk_detections(draw):
    """SDK-valid detections: boxes inside the image (the SDK clips), unique confidences."""
    count = draw(st.integers(min_value=0, max_value=6))
    # float32 values: the pipeline holds scores in float32 tensors, so ties and limits must be
    # decided in float32, too
    confs = draw(
        st.lists(
            st.floats(min_value=SPECIESNET_FLOOR_F32, max_value=1.0, width=32),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    detections = []
    for conf in confs:
        x = draw(st.floats(min_value=0.0, max_value=1.0))
        y = draw(st.floats(min_value=0.0, max_value=1.0))
        w = draw(st.floats(min_value=0.0, max_value=1.0 - x))
        h = draw(st.floats(min_value=0.0, max_value=1.0 - y))
        label = draw(st.sampled_from(["animal", "human", "vehicle"]))
        detections.append(_sdk_detection(label, conf, [x, y, w, h]))
    return detections


@PROPERTY_SETTINGS
@given(detections=sdk_detections(), conf_limit=st.floats(min_value=SPECIESNET_FLOOR_F32, max_value=1.0, width=32))
def test_speciesnet_returns_labelled_boxes_as_xyxy(colour_images, detections, conf_limit):
    """All boxes >= conf_limit survive with their SDK label, converted from xywh to xyxy,
    sorted by score, and within the image."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("ml_pipeline.models.model.SpeciesNetDetector", make_speciesnet_stub(detections))
        model = SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=conf_limit)
        scores, result = model.predict(colour_images[0])

    expected = sorted((d for d in detections if d["conf"] >= conf_limit), key=lambda d: d["conf"], reverse=True)
    assert scores.tolist() == pytest.approx([d["conf"] for d in expected])
    assert [d.detected_class for d in result] == [d["label"] for d in expected]
    for detection, sdk in zip(result, expected):
        x, y, w, h = sdk["bbox"]
        box = detection.bbox
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == pytest.approx((x, y, x + w, y + h))
        assert -1e-9 <= box.x_min <= box.x_max <= 1.0 + 1e-9
        assert -1e-9 <= box.y_min <= box.y_max <= 1.0 + 1e-9
        assert detection.classifications == []


@pytest.fixture
def speciesnet_stub(monkeypatch):
    """Install a SpeciesNet stub with one detection of each label."""
    stub = make_speciesnet_stub(
        [
            _sdk_detection("animal", 0.8, [0.1, 0.2, 0.3, 0.4]),
            _sdk_detection("human", 0.9, [0.5, 0.5, 0.2, 0.2]),
            _sdk_detection("vehicle", 0.3, [0.0, 0.0, 0.1, 0.1]),
        ]
    )
    monkeypatch.setattr("ml_pipeline.models.model.SpeciesNetDetector", stub)
    return stub


def test_speciesnet_label_set():
    """SpeciesNet's detector emits exactly these labels."""
    assert tuple(SpeciesnetDetector.LABELS) == ("animal", "human", "vehicle")


def test_speciesnet_keeps_human_and_vehicle(speciesnet_stub, colour_images):
    """Humans and vehicles are returned with their SDK label, not dropped."""
    model = SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1)
    scores, detections = model.predict(colour_images[0])
    assert [d.detected_class for d in detections] == ["human", "animal", "vehicle"]
    assert scores.tolist() == pytest.approx([0.9, 0.8, 0.3])
    _assert_ordinary_tensor(scores)


def test_speciesnet_label_map(speciesnet_stub, colour_images):
    """label_map renames SDK labels like any other detector's."""
    model = SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1, label_map={"human": "person"})
    _, detections = model.predict(colour_images[0])
    assert [d.detected_class for d in detections] == ["person", "animal", "vehicle"]


def test_speciesnet_label_map_with_unknown_label_raises(speciesnet_stub):
    """Only animal, human and vehicle can be mapped."""
    with pytest.raises(ValueError, match="person"):
        SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1, label_map={"person": "human"})


def test_speciesnet_gets_a_placeholder_filepath(speciesnet_stub, colour_images):
    """Models receive no paths; the SDK, which only echoes its filepath, gets the placeholder."""
    model = SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1)
    model.predict(colour_images[0])
    model.predict(colour_images[1])
    expected = SpeciesnetDetector.SDK_FILEPATH_PLACEHOLDER
    assert speciesnet_stub.instances[-1].filepaths == [expected, expected]


def test_speciesnet_defaults_to_v403b(speciesnet_stub):
    """Without model_path the pinned SpeciesNet v4.0.3b Kaggle model is requested."""
    SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1)
    assert speciesnet_stub.instances[-1].model_name == SPECIESNET_V403B


@pytest.mark.parametrize("model_path", ["./models/speciesnet/", "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"])
def test_speciesnet_passes_model_path_unchanged(speciesnet_stub, model_path):
    """Local folders and Kaggle ids reach the SDK exactly as configured."""
    SpeciesnetDetector(info=INFO, model_path=model_path, preprocessing={}, conf_limit=0.1)
    assert speciesnet_stub.instances[-1].model_name == model_path


def test_speciesnet_runs_on_cpu(speciesnet_stub):
    """The SDK picks its own device (the stub pretends CUDA); the pipeline forces CPU."""
    SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1)
    detector = speciesnet_stub.instances[-1]
    assert detector.device == "cpu"
    assert detector.model.moved
    assert all(p.device.type == "cpu" for p in detector.model.parameters())


def test_speciesnet_hub_download_not_implemented(speciesnet_stub, hub_stub):
    """hf_repo/hf_filename fetch one file, but a SpeciesNet model is a folder."""
    with pytest.raises(NotImplementedError):
        SpeciesnetDetector(
            info=INFO, hf_repo="org/speciesnet", hf_filename="model.pt", preprocessing={}, conf_limit=0.1
        )
    assert hub_stub == []


@pytest.mark.parametrize("bad_args", [{"preprocessing": {"resize": {"size": [16, 16]}}}, {"conf_limit": 0.005}])
def test_speciesnet_invalid_settings_raise(speciesnet_stub, bad_args):
    """SpeciesNet preprocesses itself and has a 0.01 score floor."""
    kwargs = {"info": INFO, "preprocessing": {}, "conf_limit": 0.1} | bad_args
    with pytest.raises(ValueError):
        SpeciesnetDetector(**kwargs)
    assert speciesnet_stub.instances == []


def test_speciesnet_accepts_its_score_floor(speciesnet_stub):
    """conf_limit 0.01, the SDK's floor itself, is valid."""
    SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.01)
    assert len(speciesnet_stub.instances) == 1


@pytest.mark.parametrize("preprocessing", [3, ["resize"]])
def test_speciesnet_non_mapping_preprocessing_raises(speciesnet_stub, preprocessing):
    """A preprocessing node that is not a mapping is a TypeError, as for every model."""
    with pytest.raises(TypeError):
        SpeciesnetDetector(info=INFO, preprocessing=preprocessing, conf_limit=0.1)
    assert speciesnet_stub.instances == []


@pytest.mark.parametrize("conf_limit", ["0.5", None])
def test_speciesnet_non_numeric_conf_limit_raises(speciesnet_stub, conf_limit):
    """A non-numeric conf_limit gets the explicit TypeError, not a comparison error."""
    with pytest.raises(TypeError, match="conf_limit"):
        SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=conf_limit)


@pytest.mark.parametrize("preprocessing", [{}, None])
def test_speciesnet_accepts_empty_preprocessing(speciesnet_stub, preprocessing):
    """An empty preprocessing node, `{}` or YAML null, is accepted."""
    SpeciesnetDetector(info=INFO, preprocessing=preprocessing, conf_limit=0.1)
    assert len(speciesnet_stub.instances) == 1


def test_speciesnet_weights_name_the_default_model(speciesnet_stub):
    """Without model_path, the weights are the pinned default Kaggle model."""
    assert SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1).weights == SPECIESNET_V403B


# ------------------------------------------------------- SpeciesNet classifier (stub)

# SDK label lines: uuid;class;order;family;genus;species;common name. As in the real label file,
# two labels (u0, u3) differ only in their uuid, and one (u4) has no common name.
SPECIESNET_LABELS = [
    "u0;mammalia;rodentia;sciuridae;;;sciuridae family",
    "u1;mammalia;rodentia;muridae;rattus;norvegicus;brown rat",
    "u2;;;;;;blank",
    "u3;mammalia;rodentia;sciuridae;;;sciuridae family",
    "u4;aves;passeriformes;;;;",
]
SCIURIDAE = Taxon(class_name="mammalia", order="rodentia", family="sciuridae", common_name="sciuridae family")
SPECIESNET_TAXA = [
    SCIURIDAE,
    Taxon(
        class_name="mammalia",
        order="rodentia",
        family="muridae",
        genus="rattus",
        species="norvegicus",
        common_name="brown rat",
    ),
    Taxon(common_name="blank"),
    SCIURIDAE,
    Taxon(class_name="aves", order="passeriformes"),
]
# by label index; the SDK reports them sorted by score (1, 3, 0, 4, 2), so index order must be restored
SPECIESNET_SCORES = [0.15, 0.4, 0.05, 0.3, 0.1]


def make_speciesnet_classifier_stub(
    model_type: str = "always_crop", labels: list[str] | None = None, scores: list[float] | None = None
):
    """Create a stand-in for `speciesnet.classifier.SpeciesNetClassifier`.

    It mimics the SDK surface used by the pipeline: construction from a model name with a
    `device` argument, `model_info.type_`, the `labels` index -> label line mapping,
    `preprocess(PIL image, bboxes)` returning one marker per box, and
    `predict(filepath, preprocessed, num_classes)` returning SDK-shaped classifications (labels
    and scores sorted by score). Instances record the bboxes, filepaths and num_classes they got.
    `labels` and `scores` default to SPECIESNET_LABELS / SPECIESNET_SCORES.
    """
    labels = SPECIESNET_LABELS if labels is None else labels
    scores = SPECIESNET_SCORES if scores is None else scores
    instances = []

    class Preprocessed:
        """Marker for one crop produced by the stub's `preprocess`."""

        def __init__(self, image, bbox):
            self.image = image
            self.bbox = bbox

    class StubModelInfo:
        """The SDK's ModelInfo, reduced to the model type."""

        type_ = model_type

    class StubSpeciesNetClassifier:
        """Stand-in for the SDK classifier component."""

        def __init__(self, model_name, target_species_txt=None, device=None, force_model_download=False):
            self.model_name = model_name
            self.device = device
            self.model_info = StubModelInfo()
            self.labels = dict(enumerate(labels))
            self.bboxes = []
            self.filepaths = []
            self.num_classes = []
            instances.append(self)

        def preprocess(self, img, bboxes=None, resize=True):
            """Accept only RGB PIL images; return one crop per box, as `always_crop` does."""
            assert isinstance(img, Image.Image) and img.mode == "RGB"
            self.bboxes.append(list(bboxes or []))
            return [Preprocessed(img, bbox) for bbox in bboxes or []]

        def predict(self, filepath, imgs, num_classes=5):
            """Return the scores of every crop, sorted by score like the SDK."""
            assert imgs and all(isinstance(img, Preprocessed) for img in imgs)
            self.filepaths.append(filepath)
            self.num_classes.append(num_classes)
            order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
            if num_classes is not None:
                order = order[:num_classes]
            classification = {"classes": [labels[i] for i in order], "scores": [scores[i] for i in order]}
            return {"filepath": filepath, "classifications_list": [classification for _ in imgs]}

    StubSpeciesNetClassifier.instances = instances
    return StubSpeciesNetClassifier


@pytest.fixture
def speciesnet_classifier_stub(monkeypatch):
    """Install a SpeciesNet classifier stub for an `always_crop` model."""
    stub = make_speciesnet_classifier_stub()
    monkeypatch.setattr("ml_pipeline.models.model.SpeciesNetClassifier", stub)
    return stub


def make_speciesnet_classifier(**overrides) -> SpeciesnetClassifier:
    """Build a SpeciesnetClassifier with test defaults, overridable per test."""
    kwargs = {"info": INFO, "preprocessing": {}, "conf_limit": 0.1} | overrides
    return SpeciesnetClassifier(**kwargs)


def test_speciesnet_classifier_taxonomy_comes_from_the_labels(speciesnet_classifier_stub):
    """Each label line becomes a Taxon; empty fields (no genus, no common name, no taxonomy at
    all) are None."""
    assert make_speciesnet_classifier().classes == SPECIESNET_TAXA


def test_speciesnet_classifier_returns_probabilities_in_label_order(speciesnet_classifier_stub, colour_images):
    """The probability vector follows the label index, not the SDK's score order, also for labels
    that differ only in their uuid; the payload holds the classes >= conf_limit, sorted, with
    their taxonomy."""
    probs, classifications = make_speciesnet_classifier(conf_limit=0.1).predict(colour_images[0])

    assert probs.tolist() == pytest.approx(SPECIESNET_SCORES)
    assert [c.confidence for c in classifications] == pytest.approx([0.4, 0.3, 0.15, 0.1])
    assert [c.model_dump(exclude={"confidence"}) for c in classifications] == [
        SPECIESNET_TAXA[1].model_dump(),
        SCIURIDAE.model_dump(),
        SCIURIDAE.model_dump(),
        SPECIESNET_TAXA[4].model_dump(),
    ]
    _assert_ordinary_tensor(probs)


@st.composite
def labelled_scores(draw):
    """SDK label lines (unique uuids, taxa repeating so some differ only by uuid) and float32 scores."""
    count = draw(st.integers(min_value=1, max_value=8))
    labels = [f"u{i};mammalia;rodentia;muridae;genus{i % 3};;name{i % 3}" for i in range(count)]
    scores = draw(st.lists(st.floats(0.0, 1.0, width=32), min_size=count, max_size=count))
    return labels, scores


@PROPERTY_SETTINGS
@given(case=labelled_scores(), conf_limit=st.floats(0.0, 1.0, width=32))
def test_speciesnet_classifier_reorders_and_filters_any_scores(colour_images, case, conf_limit):
    """For any labels and scores (ties included): the probability vector is the scores in label
    order, and the classifications are exactly the scores >= conf_limit, descending, each with
    the taxonomy of its own label. The SDK's softmax scores are taken as given, so the drawn
    scores need not sum to 1."""
    labels, scores = case
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "ml_pipeline.models.model.SpeciesNetClassifier",
            make_speciesnet_classifier_stub(labels=labels, scores=scores),
        )
        probs, classifications = make_speciesnet_classifier(conf_limit=conf_limit).predict(colour_images[0])

    assert probs.tolist() == pytest.approx(scores)
    expected = sorted((score for score in scores if score >= conf_limit), reverse=True)
    assert [c.confidence for c in classifications] == pytest.approx(expected)
    # taxonomy per label (as a multiset: the order between tied scores is open)
    expected_pairs = sorted(
        (label.split(";")[4], score) for label, score in zip(labels, scores) if score >= conf_limit
    )
    assert sorted((c.genus, c.confidence) for c in classifications) == pytest.approx(expected_pairs)


def test_speciesnet_classifier_asks_for_all_classes_of_the_whole_crop(speciesnet_classifier_stub, colour_images):
    """The SDK gets the whole crop as its only box, every class's score, and the placeholder
    filepath (models receive no paths)."""
    make_speciesnet_classifier().predict(colour_images[0])
    sdk = speciesnet_classifier_stub.instances[-1]

    ((bbox,),) = sdk.bboxes
    assert (bbox.xmin, bbox.ymin, bbox.width, bbox.height) == (0.0, 0.0, 1.0, 1.0)
    assert sdk.num_classes == [None]
    assert sdk.filepaths == [SpeciesnetClassifier.SDK_FILEPATH_PLACEHOLDER]


def test_speciesnet_classifier_runs_on_cpu_with_the_default_model(speciesnet_classifier_stub):
    """The SDK gets device="cpu" and, without model_path, the pinned v4.0.3a crop classifier."""
    model = make_speciesnet_classifier()
    sdk = speciesnet_classifier_stub.instances[-1]
    assert sdk.device == "cpu"
    assert sdk.model_name == SPECIESNET_V403A
    assert model.weights == SPECIESNET_V403A


def test_speciesnet_classifier_passes_model_path_unchanged(speciesnet_classifier_stub):
    """A configured model folder or Kaggle id reaches the SDK as given."""
    model = make_speciesnet_classifier(model_path="./models/speciesnet/")
    assert speciesnet_classifier_stub.instances[-1].model_name == "./models/speciesnet/"
    assert model.weights == "./models/speciesnet/"


def test_speciesnet_classifier_rejects_malformed_label_lines(monkeypatch):
    """A label file whose lines are not `uuid;class;order;family;genus;species;common name` is an error."""
    monkeypatch.setattr(
        "ml_pipeline.models.model.SpeciesNetClassifier",
        make_speciesnet_classifier_stub(labels=["u0;mammalia;rodentia"], scores=[1.0]),
    )
    with pytest.raises(ValueError, match="7"):
        make_speciesnet_classifier()


def test_speciesnet_classifier_rejects_duplicate_label_lines(monkeypatch):
    """Scores are matched to classes by label line, so lines must be unique."""
    duplicated = "u0;mammalia;rodentia;muridae;rattus;rattus;house rat"
    monkeypatch.setattr(
        "ml_pipeline.models.model.SpeciesNetClassifier",
        make_speciesnet_classifier_stub(labels=[duplicated, duplicated], scores=[0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="unique"):
        make_speciesnet_classifier()


def _incomplete_info_json(*args, **kwargs):
    """Stand-in for an SDK constructor reading a local model folder whose info.json lacks a key."""
    raise KeyError("classifier_labels")


@pytest.mark.parametrize(
    ("sdk_class", "make_model"),
    [
        ("SpeciesNetDetector", lambda: SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.1)),
        ("SpeciesNetClassifier", make_speciesnet_classifier),
    ],
)
def test_speciesnet_with_incomplete_info_json_raises_value_error(monkeypatch, sdk_class, make_model):
    """A SpeciesNet model folder with an incomplete info.json is an invalid config (ValueError)."""
    monkeypatch.setattr(f"ml_pipeline.models.model.{sdk_class}", _incomplete_info_json)
    with pytest.raises(ValueError, match="info.json"):
        make_model()


def test_speciesnet_classifier_rejects_whole_image_models(monkeypatch):
    """A `full_image` model (v4.0.3b) expects whole images, not crops: rejected at construction."""
    monkeypatch.setattr("ml_pipeline.models.model.SpeciesNetClassifier", make_speciesnet_classifier_stub("full_image"))
    with pytest.raises(ValueError, match="full_image"):
        make_speciesnet_classifier(model_path="kaggle:google/speciesnet/pyTorch/v4.0.3b/1")


def test_speciesnet_classifier_hub_download_not_implemented(speciesnet_classifier_stub, hub_stub):
    """hf_repo/hf_filename fetch one file, but a SpeciesNet model is a folder."""
    with pytest.raises(NotImplementedError):
        make_speciesnet_classifier(hf_repo="org/speciesnet", hf_filename="model.pt")
    assert hub_stub == []


@pytest.mark.parametrize(
    ("bad_args", "error"),
    [
        ({"preprocessing": {"resize": {"size": [16, 16]}}}, ValueError),
        ({"classes": TAXA}, TypeError),
        ({"conf_limit": 1.5}, ValueError),
    ],
)
def test_speciesnet_classifier_invalid_settings_raise(speciesnet_classifier_stub, bad_args, error):
    """SpeciesNet preprocesses itself and brings its own taxonomy; checked before loading."""
    with pytest.raises(error):
        make_speciesnet_classifier(**bad_args)
    assert speciesnet_classifier_stub.instances == []


def test_load_model_builds_the_speciesnet_classifier(speciesnet_classifier_stub, tmp_path):
    """`type: SpeciesnetClassifier` builds the SDK-backed classifier."""
    config = {"type": "SpeciesnetClassifier", "info": INFO, "preprocessing": {}, "conf_limit": 0.05}
    assert type(load_model(_write_config(tmp_path / "speciesnet_cls.yaml", config))) is SpeciesnetClassifier


# -------------------------------------------------------- SpeciesNet (real, opt-in)


@pytest.mark.speciesnet_integration
def test_real_speciesnet_detects_rodent_and_people(ultralytics_images):
    """Real SpeciesNet inference. Run with RUN_SPECIESNET_INTEGRATION=1.

    SPECIESNET_MODEL may name a local model folder or Kaggle id; otherwise the pinned
    default (v4.0.3b) is loaded from the Kaggle cache, downloading it on first use.
    """
    if os.environ.get("RUN_SPECIESNET_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPECIESNET_INTEGRATION=1 to run real SpeciesNet inference")
    model_kwargs = {}
    if "SPECIESNET_MODEL" in os.environ:
        model_kwargs["model_path"] = os.environ["SPECIESNET_MODEL"]

    model = SpeciesnetDetector(info=INFO, preprocessing={}, conf_limit=0.5, **model_kwargs)
    _, rodent = model.predict(_open_rgb(Path(__file__).parent / "data" / "rodent.jpg"))
    _, people = model.predict(_open_rgb(ultralytics_images["zidane"]))

    assert "animal" in {d.detected_class for d in rodent}
    labels = {d.detected_class for d in people}
    assert "animal" not in labels
    assert "human" in labels
