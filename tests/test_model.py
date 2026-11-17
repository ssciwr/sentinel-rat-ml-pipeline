"""Behavioural tests for the model handling module (`ml_pipeline.models.model`).

Test layout follows planning/plan_model_handling.md, section 4:
- torch models: real pickled `.pt` files of tiny modules (tests/tiny_models.py)
- YOLO: the pretrained default models, downloaded once into the pytest cache
- SpeciesNet: an SDK stub (CI must not download the ~500 MB Kaggle model), plus an opt-in
  real-model test (`RUN_SPECIESNET_INTEGRATION=1`)
"""

import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageOps
from pydantic import ValidationError
from torchvision.transforms import v2

from ml_pipeline.models.model import (
    SpeciesnetDetector,
    TorchClassifier,
    TorchDetector,
    YoloClassifier,
    YoloDetector,
    build_transform,
    load_image,
)
from ml_pipeline.schemas.analysis import BoundingBox, Classification, Taxon
from tests import tiny_models

PROPERTY_SETTINGS = settings(deadline=None, max_examples=25)

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

DETECTOR_ROWS = [
    [0.1, 0.2, 0.4, 0.5, 0.6],
    [0.2, 0.1, 0.9, 0.8, 0.95],
    [0.3, 0.3, 0.6, 0.6, 0.05],
    [-1.0, -1.0, -1.0, -1.0, 0.0],  # padding: score 0, arbitrary (even invalid) box
]

SPECIESNET_V403B = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
# smallest float32 >= 0.01 (SpeciesNet's score floor); 0.01 itself is not representable in float32
SPECIESNET_FLOOR_F32 = float(np.nextafter(np.float32(0.01), np.float32(1.0)))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Session-wide directory for generated images and pickled models."""
    return tmp_path_factory.mktemp("model_data")


@pytest.fixture(scope="session")
def colour_images(data_dir) -> list[Path]:
    """Six 32x32 RGB images of distinct uniform colours, so model outputs differ per image."""
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (40, 40, 40)]
    paths = []
    for index, colour in enumerate(colours):
        path = data_dir / f"colour_{index}.png"
        Image.new("RGB", (32, 32), colour).save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def odd_size_image(data_dir) -> Path:
    """An image whose size differs from the colour images."""
    path = data_dir / "odd_size.png"
    Image.new("RGB", (48, 40), (10, 200, 30)).save(path)
    return path


def _save_module(module: object, path: Path) -> Path:
    """Pickle a whole module (or any object) with torch.save, the format the pipeline loads."""
    torch.save(module, path)
    return path


def _seeded(seed: int, build):
    """Build a module under a fixed seed without changing the global RNG state of other tests."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return build()


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


PROBE_WIDTH = 8
PROBE_TAXA = [{"genus": f"Genus{i}", "species": f"species{i}"} for i in range(PROBE_WIDTH)]


@pytest.fixture(scope="session")
def batch_probe_pt(data_dir) -> Path:
    """Tiny classifier whose output reveals the batch size it was called with."""
    return _save_module(tiny_models.BatchSizeProbe(PROBE_WIDTH), data_dir / "batch_probe.pt")


def make_torch_classifier(model_path: Path, **overrides) -> TorchClassifier:
    """Build a TorchClassifier with test defaults, overridable per test."""
    kwargs = {
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
    kwargs = {"model_path": model_path, "preprocessing": {}, "conf_limit": 0.0}
    kwargs.update(overrides)
    return TorchDetector(**kwargs)


def species_index(classification: Classification) -> int:
    """Return the TAXA index of a classification (species names are unique in TAXA)."""
    return [taxon["species"] for taxon in TAXA].index(classification.species)


# ------------------------------------------------------------------- torch classifier


def test_torch_classifier_runs_in_eval_mode(logits_classifier_pt, colour_images):
    """Dropout is inactive: repeated predictions are identical."""
    model = make_torch_classifier(logits_classifier_pt)
    first, _ = model.predict(colour_images)
    second, _ = model.predict(colour_images)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.fixture(scope="session")
def single_image_references(logits_classifier_pt, colour_detector_pt, colour_images):
    """Per-image results of one-image predict calls, as the reference for batched calls."""
    classifier = make_torch_classifier(logits_classifier_pt)
    detector = make_torch_detector(colour_detector_pt)
    classifier_probs = [classifier.predict([path])[0][0] for path in colour_images]
    detector_results = [detector.predict([path]) for path in colour_images]
    detector_scores = [scores[0] for scores, _ in detector_results]
    detector_boxes = [boxes[0] for _, boxes in detector_results]
    return classifier_probs, detector_scores, detector_boxes


def _box_coords(boxes: list[BoundingBox]) -> list[float]:
    """Corners of all boxes as one flat list (pytest.approx does not compare nested tuples)."""
    return [coord for b in boxes for coord in (b.x_min, b.y_min, b.x_max, b.y_max)]


@PROPERTY_SETTINGS
@given(
    indices=st.lists(st.integers(min_value=0, max_value=5), max_size=7),
    batch_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    as_str=st.booleans(),
)
def test_torch_predict_returns_one_entry_per_path_in_order(
    logits_classifier_pt, colour_detector_pt, colour_images, single_image_references,
    indices, batch_size, as_str,
):
    """Any number of paths (str or Path) and any batch size (incl. None and a short last
    chunk) give per-path results in input order that match single-image predictions,
    for classifiers and for detectors whose rows differ per image."""
    expected_probs, expected_scores, expected_boxes = single_image_references
    paths = [str(colour_images[i]) if as_str else colour_images[i] for i in indices]

    classifier = make_torch_classifier(logits_classifier_pt, batch_size=batch_size)
    probabilities, payload = classifier.predict(paths)
    assert len(probabilities) == len(payload) == len(indices)
    for index, probs in zip(indices, probabilities):
        assert torch.allclose(probs, expected_probs[index], atol=1e-6)

    detector = make_torch_detector(colour_detector_pt, batch_size=batch_size)
    scores, boxes = detector.predict(paths)
    assert len(scores) == len(boxes) == len(indices)
    for index, image_scores, image_boxes in zip(indices, scores, boxes):
        assert torch.allclose(image_scores, expected_scores[index], atol=1e-6)
        assert _box_coords(image_boxes) == pytest.approx(_box_coords(expected_boxes[index]), abs=1e-6)


@PROPERTY_SETTINGS
@given(
    count=st.integers(min_value=1, max_value=7),
    batch_size=st.one_of(st.none(), st.integers(min_value=1, max_value=PROBE_WIDTH)),
)
def test_torch_batches_honour_batch_size(batch_probe_pt, colour_images, count, batch_size):
    """The module sees chunks of batch_size images (the last one may be shorter); without
    batch_size it sees all images of the predict call at once."""
    model = TorchClassifier(
        model_path=batch_probe_pt,
        preprocessing={},
        conf_limit=0.5,
        classes=PROBE_TAXA,
        apply_softmax=False,
        batch_size=batch_size,
    )
    probabilities, _ = model.predict([colour_images[i % len(colour_images)] for i in range(count)])
    observed = [int(probs.argmax()) + 1 for probs in probabilities]

    if batch_size is None:
        expected = [count] * count
    else:
        full_chunks, remainder = divmod(count, batch_size)
        expected = [batch_size] * (full_chunks * batch_size) + [remainder] * remainder
    assert observed == expected


@PROPERTY_SETTINGS
@given(conf_limit=st.floats(min_value=0.0, max_value=1.0, width=32))
def test_torch_classifier_payload_invariants(logits_classifier_pt, colour_images, conf_limit):
    """Payload: sorted descending, all >= conf_limit, taxonomy matches the output index,
    complete w.r.t. conf_limit; softmax-ed probability vectors sum to 1."""
    model = make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit)
    probabilities, payload = model.predict(colour_images)

    for probs, classifications in zip(probabilities, payload):
        assert probs.shape == (len(TAXA),)
        assert probs.device.type == "cpu"
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        confidences = [c.confidence for c in classifications]
        assert confidences == sorted(confidences, reverse=True)
        assert all(c >= conf_limit for c in confidences)
        assert len(classifications) == int((probs >= conf_limit).sum())
        for classification in classifications:
            assert isinstance(classification, Classification)
            index = species_index(classification)
            assert classification.confidence == pytest.approx(float(probs[index]))
            assert classification.model_dump(exclude={"confidence"}) == TAXA[index]


def test_torch_classifier_accepts_probabilities_without_softmax(
    probability_classifier_pt, colour_images
):
    """With apply_softmax false the module's probabilities are used as they are."""
    model = make_torch_classifier(probability_classifier_pt, apply_softmax=False)
    probabilities, _ = model.predict(colour_images)
    raw = torch.load(probability_classifier_pt, weights_only=False).eval()
    for path, probs in zip(colour_images, probabilities):
        image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(load_image(path))
        with torch.inference_mode():
            expected = raw(image.unsqueeze(0))[0]
        assert torch.allclose(probs, expected, atol=1e-6)


def test_torch_preprocessing_runs_after_tensor_conversion(probability_classifier_pt, colour_images):
    """Configured transforms act on the float [0, 1] tensor: a `normalize` config equals
    feeding the raw module a hand-normalised input."""
    normalise = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    model = make_torch_classifier(
        probability_classifier_pt, apply_softmax=False, preprocessing={"normalize": normalise}
    )
    probabilities, _ = model.predict(colour_images)
    raw = torch.load(probability_classifier_pt, weights_only=False).eval()
    to_input = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(**normalise)]
    )
    for path, probs in zip(colour_images, probabilities):
        with torch.inference_mode():
            expected = raw(to_input(load_image(path)).unsqueeze(0))[0]
        assert torch.allclose(probs, expected, atol=1e-6)


def test_torch_classifier_taxonomy_length_mismatch_raises(logits_classifier_pt, colour_images):
    """A model output width different from the number of configured classes is rejected."""
    model = make_torch_classifier(logits_classifier_pt, classes=TAXA[:2])
    with pytest.raises(ValueError):
        model.predict(colour_images[:1])


def test_torch_images_of_different_size_without_resize_raise(
    logits_classifier_pt, colour_images, odd_size_image
):
    """Images that cannot be stacked into one batch give a clear ValueError."""
    model = make_torch_classifier(logits_classifier_pt)
    with pytest.raises(ValueError):
        model.predict([colour_images[0], odd_size_image])


def test_torch_images_of_different_size_with_resize_work(
    logits_classifier_pt, colour_images, odd_size_image
):
    """A resize in the preprocessing node makes differently sized images stackable."""
    model = make_torch_classifier(
        logits_classifier_pt, preprocessing={"resize": {"size": [16, 16], "antialias": True}}
    )
    probabilities, payload = model.predict([colour_images[0], odd_size_image])
    assert len(probabilities) == len(payload) == 2


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
    sys.modules.pop("vanishing_models")
    with pytest.raises(ImportError, match="importable"):
        make_torch_classifier(path)


def test_torch_pickled_non_module_raises(data_dir):
    """A .pt file that does not contain an nn.Module is rejected at construction."""
    path = _save_module({"weights": torch.zeros(1)}, data_dir / "not_a_module.pt")
    with pytest.raises(TypeError):
        make_torch_classifier(path)


# --------------------------------------------------------------------- torch detector


def test_torch_detector_payload_and_padding(fixed_detector_pt, colour_images):
    """Boxes are BoundingBoxes aligned with sorted scores; zero-score padding rows are
    dropped even at conf_limit 0 and never raise, although their boxes are invalid."""
    model = make_torch_detector(fixed_detector_pt, conf_limit=0.0)
    probabilities, payload = model.predict(colour_images[:2])

    assert len(probabilities) == len(payload) == 2
    for scores, boxes in zip(probabilities, payload):
        assert scores.device.type == "cpu"
        assert scores.tolist() == pytest.approx([0.95, 0.6, 0.05])
        assert all(isinstance(box, BoundingBox) for box in boxes)
        assert _box_coords(boxes) == pytest.approx(
            [0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.4, 0.5, 0.3, 0.3, 0.6, 0.6]
        )


@PROPERTY_SETTINGS
@given(conf_limit=st.floats(min_value=0.0, max_value=1.0, width=32))
def test_torch_detector_conf_limit(fixed_detector_pt, colour_images, conf_limit):
    """Only real rows with score >= conf_limit are kept, sorted descending, scores aligned."""
    model = make_torch_detector(fixed_detector_pt, conf_limit=conf_limit)
    probabilities, payload = model.predict(colour_images[:1])
    scores32 = torch.tensor(DETECTOR_ROWS)[:, 4]  # float32, like the pipeline's comparison
    kept = scores32[(scores32 > 0) & (scores32 >= conf_limit)]
    expected = sorted(kept.tolist(), reverse=True)
    assert probabilities[0].tolist() == pytest.approx(expected)
    assert len(payload[0]) == len(expected)


def test_torch_detector_number_of_rows_may_vary_between_calls(data_dir, colour_images):
    """K may differ per call; each call's rows end up with the right image."""
    row_sets = [
        [[0.1, 0.1, 0.2, 0.2, 0.9]],
        [[0.1, 0.1, 0.2, 0.2, 0.9], [0.3, 0.3, 0.4, 0.4, 0.8]],
        [[0.1, 0.1, 0.2, 0.2, 0.9], [0.3, 0.3, 0.4, 0.4, 0.8], [0.5, 0.5, 0.6, 0.6, 0.7]],
    ]
    path = _save_module(tiny_models.VaryingRowsDetector(row_sets), data_dir / "varying.pt")
    model = make_torch_detector(path, batch_size=1)
    _, payload = model.predict(colour_images[:3])
    assert [len(boxes) for boxes in payload] == [1, 2, 3]


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
        model.predict(colour_images[:2])


@pytest.mark.parametrize(
    "mode",
    [
        "det_tuple",
        "det_rank2",
        "det_last_dim",
        "det_batch",
        "det_score_range",
        "det_score_nan",
        "det_box_range",
        "det_box_inverted",
    ],
)
def test_torch_detector_contract_violations_raise(data_dir, colour_images, mode):
    """Every violation of the detector output contract raises a ValueError."""
    path = _save_module(tiny_models.ContractBreaker(mode), data_dir / f"det_{mode}.pt")
    model = make_torch_detector(path, conf_limit=0.25)
    with pytest.raises(ValueError):
        model.predict(colour_images[:2])


# ------------------------------------------------------------ predict contract (3.1)


def test_predict_on_empty_list_returns_empty_results(logits_classifier_pt, fixed_detector_pt):
    """No paths, no results, for classifiers and detectors alike."""
    assert make_torch_classifier(logits_classifier_pt).predict([]) == ([], [])
    assert make_torch_detector(fixed_detector_pt).predict([]) == ([], [])


@pytest.mark.parametrize("single_path", ["image.jpg", Path("image.jpg")])
def test_predict_rejects_a_single_path(logits_classifier_pt, single_path):
    """predict takes a list of paths; a bare str/Path is an error, not a sequence of characters."""
    with pytest.raises(TypeError):
        make_torch_classifier(logits_classifier_pt).predict(single_path)


def test_torch_preprocessing_must_produce_tensors(logits_classifier_pt, colour_images):
    """Torch models need tensors; a config that turns them back into PIL images is a TypeError."""
    model = make_torch_classifier(logits_classifier_pt, preprocessing={"to_pil_image": {}})
    with pytest.raises(TypeError):
        model.predict(colour_images[:1])


def test_predict_on_missing_file_raises(logits_classifier_pt, tmp_path):
    """A missing image is an error, not a silently skipped entry."""
    model = make_torch_classifier(logits_classifier_pt)
    with pytest.raises(FileNotFoundError):
        model.predict([tmp_path / "does_not_exist.jpg"])


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
    ("resize", {"size": [16, 16], "antialias": True}, v2.Resize),
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


@pytest.mark.parametrize("name", ["no_such_transform", "functional", "query_size"])
def test_build_transform_rejects_unknown_and_non_transform_names(name):
    """Only v2.Transform subclasses resolve; anything else is a ValueError."""
    with pytest.raises(ValueError):
        build_transform({name: {}})


@pytest.mark.parametrize("kwargs", [16, [16, 16], "bilinear"])
def test_build_transform_rejects_non_mapping_kwargs(kwargs):
    """Transform kwargs must be a mapping."""
    with pytest.raises(TypeError):
        build_transform({"resize": kwargs})


def test_build_transform_rejects_abstract_base_transform():
    """The abstract v2.Transform base is not a usable transform."""
    with pytest.raises(ValueError):
        build_transform({"transform": {}})


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


# --------------------------------------------------------- config and __init__ checks


def test_missing_preprocessing_raises(logits_classifier_pt):
    """The preprocessing node is required."""
    with pytest.raises(TypeError):
        TorchClassifier(model_path=logits_classifier_pt, conf_limit=0.1, classes=TAXA, apply_softmax=True)


def test_model_path_and_hf_repo_together_raise(logits_classifier_pt, hub_stub):
    """Weights come from exactly one source."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, hf_repo="org/repo", hf_filename="m.pt")
    assert hub_stub == []


@pytest.mark.parametrize("model_class", [TorchClassifier, TorchDetector, YoloClassifier, YoloDetector])
def test_neither_model_path_nor_hf_repo_raises(model_class):
    """Without a default (only SpeciesNet has one), a weights source is required."""
    kwargs = {"preprocessing": {}, "conf_limit": 0.1}
    if model_class in (TorchClassifier, YoloClassifier):
        kwargs["classes"] = TAXA
    if model_class is TorchClassifier:
        kwargs["apply_softmax"] = True
    with pytest.raises(ValueError):
        model_class(**kwargs)


def test_hf_repo_without_filename_raises(hub_stub):
    """A Hub download needs the file name."""
    with pytest.raises(ValueError):
        TorchClassifier(hf_repo="org/repo", preprocessing={}, conf_limit=0.1, classes=TAXA, apply_softmax=True)
    assert hub_stub == []


@pytest.mark.parametrize("conf_limit", [-0.1, 1.1, float("nan")])
def test_conf_limit_outside_unit_interval_raises(logits_classifier_pt, conf_limit):
    """conf_limit must lie in [0, 1]."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, conf_limit=conf_limit)


@pytest.mark.parametrize("stray", [{"hf_filename": "m.pt"}, {"hf_revision": "v1"}])
def test_hub_arguments_without_hf_repo_raise(logits_classifier_pt, hub_stub, stray):
    """hf_filename / hf_revision next to a local model_path are errors, not silently ignored."""
    with pytest.raises(ValueError):
        make_torch_classifier(logits_classifier_pt, **stray)
    assert hub_stub == []


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


@pytest.mark.parametrize(("batch_size", "error"), [(0, ValueError), (-3, ValueError), (True, TypeError), (2.5, TypeError)])
def test_invalid_batch_size_raises(logits_classifier_pt, batch_size, error):
    """batch_size must be None or an int >= 1; bool and float are rejected explicitly."""
    with pytest.raises(error):
        make_torch_classifier(logits_classifier_pt, batch_size=batch_size)


def _write_config(path: Path, config) -> Path:
    """Dump a config object to a YAML file."""
    path.write_text(yaml.safe_dump(config))
    return path


def _classifier_config(model_path: Path) -> dict:
    """A complete TorchClassifier config."""
    return {
        "model_path": str(model_path),
        "preprocessing": {"resize": {"size": [16, 16], "antialias": True}},
        "conf_limit": 0.1,
        "batch_size": 2,
        "classes": TAXA,
        "apply_softmax": True,
    }


def test_from_config_matches_keyword_construction(logits_classifier_pt, colour_images, tmp_path):
    """A model built from a config file predicts exactly like one built from the same kwargs."""
    config = _classifier_config(logits_classifier_pt)
    from_file = TorchClassifier.from_config(_write_config(tmp_path / "cls.yaml", config))
    from_kwargs = TorchClassifier(**config)

    probs_file, payload_file = from_file.predict(colour_images)
    probs_kwargs, payload_kwargs = from_kwargs.predict(colour_images)
    for a, b in zip(probs_file, probs_kwargs):
        assert torch.equal(a, b)
    assert payload_file == payload_kwargs


def test_from_config_with_null_preprocessing_node(logits_classifier_pt, colour_images, tmp_path):
    """A bare `preprocessing:` (YAML null) counts as an empty node."""
    path = tmp_path / "cls.yaml"
    path.write_text(
        f"model_path: {logits_classifier_pt}\npreprocessing:\nconf_limit: 0.1\napply_softmax: true\n"
        + yaml.safe_dump({"classes": TAXA})
    )
    probabilities, _ = TorchClassifier.from_config(path).predict(colour_images[:1])
    assert len(probabilities) == 1


def test_from_config_missing_preprocessing_raises(logits_classifier_pt, tmp_path):
    """The preprocessing node is required in config files, too."""
    config = _classifier_config(logits_classifier_pt)
    del config["preprocessing"]
    with pytest.raises(TypeError):
        TorchClassifier.from_config(_write_config(tmp_path / "cls.yaml", config))


def test_from_config_unknown_key_raises(logits_classifier_pt, tmp_path):
    """Unknown config keys are errors, not silently ignored."""
    config = _classifier_config(logits_classifier_pt) | {"learning_rate": 0.1}
    with pytest.raises(TypeError):
        TorchClassifier.from_config(_write_config(tmp_path / "cls.yaml", config))


@pytest.mark.parametrize("content", [["a", "b"], "just a string", 42])
def test_from_config_non_mapping_top_level_raises(tmp_path, content):
    """A config file must contain a mapping at the top level."""
    with pytest.raises(TypeError):
        TorchClassifier.from_config(_write_config(tmp_path / "cls.yaml", content))


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


@pytest.mark.parametrize("revision", [None, "v1.2"])
def test_hf_download_is_used_and_loaded(hub_stub, colour_images, revision):
    """HF args are forwarded to hf_hub_download and the downloaded file is loaded."""
    model = TorchClassifier(
        hf_repo="org/rodent-models",
        hf_filename="weights/classifier.pt",
        hf_revision=revision,
        preprocessing={},
        conf_limit=0.1,
        classes=TAXA,
        apply_softmax=True,
    )
    assert hub_stub == [
        {"repo_id": "org/rodent-models", "filename": "weights/classifier.pt", "revision": revision}
    ]
    probabilities, _ = model.predict(colour_images[:1])
    assert probabilities[0].shape == (len(TAXA),)


@pytest.mark.parametrize(
    "bad_args",
    [
        {"preprocessing": {"no_such_transform": {}}},
        {"classes": [{"species": "rattus"}]},  # genus is required
        {"classes": [{"genus": "Rattus", "specis": "norvegicus"}]},  # misspelled key
    ],
)
def test_invalid_config_fails_before_download(hub_stub, bad_args):
    """Config errors surface before any weights are resolved or downloaded."""
    kwargs = {
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


# -------------------------------------------------------------------------- taxonomy


def test_taxon_requires_genus():
    """Taxonomy entries are validated; genus is mandatory."""
    with pytest.raises(ValidationError):
        Taxon(species="rattus")


def test_classification_json_is_unchanged_by_taxon_base():
    """Classification = Taxon + confidence; its serialised JSON keeps the previous keys,
    order and content."""
    taxon = Taxon(**TAXA[0])
    classification = Classification(**taxon.model_dump(), confidence=0.7)
    serialised = json.loads(classification.model_dump_json())
    assert list(serialised) == [
        "kingdom",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species",
        "common_name",
        "confidence",
    ]
    assert serialised == TAXA[0] | {"confidence": 0.7}


# --------------------------------------------------------- returned tensors (all types)


def _assert_ordinary_tensors(tensors: list[torch.Tensor]) -> None:
    """Returned tensors are normal CPU tensors that callers may modify in place."""
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        assert not tensor.is_inference()
        tensor.mul_(1.0)  # raises for inference tensors


@pytest.mark.parametrize("apply_softmax", [True, False])
def test_torch_classifier_returns_ordinary_tensors(probability_classifier_pt, colour_images, apply_softmax):
    """No inference-mode tensors leak out of predict, whatever the output dtype path."""
    model = make_torch_classifier(probability_classifier_pt, apply_softmax=apply_softmax)
    _assert_ordinary_tensors(model.predict(colour_images[:2])[0])


def test_torch_detector_returns_ordinary_tensors(fixed_detector_pt, colour_images):
    """No inference-mode tensors leak out of a detector's predict."""
    _assert_ordinary_tensors(make_torch_detector(fixed_detector_pt).predict(colour_images[:2])[0])


# ---------------------------------------------------------------------------- YOLO


@pytest.fixture(scope="session")
def yolo_weights(pytestconfig):
    """Return a function giving the path of a pretrained YOLO model in the pytest cache.

    Ultralytics downloads known asset names to the given path on first use, so the
    weights are fetched once and reused across runs. Without network or cache this
    fails loudly instead of skipping, so YOLO coverage cannot silently disappear.
    """
    from ultralytics import YOLO

    def weights(name: str) -> Path:
        try:
            path = pytestconfig.cache.mkdir("yolo_weights") / name
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


def _direct_yolo_scores(weights: Path, images: list[Image.Image], conf: float) -> list[list[float]]:
    """Per-image box scores of a plain ultralytics call on PIL images, all classes kept."""
    from ultralytics import YOLO

    results = YOLO(str(weights)).predict(source=images, conf=conf, device="cpu", verbose=False)
    return [sorted(r.boxes.conf.cpu().tolist(), reverse=True) for r in results]


def test_yolo_detector_returns_class_agnostic_boxes(yolo_weights, ultralytics_images):
    """The pretrained COCO detector: every box of every class is kept (same scores as a
    plain ultralytics call on the same images), results map to the right image, and the
    payload is plain BoundingBoxes aligned with sorted scores above conf_limit."""
    weights = yolo_weights("yolo26n.pt")
    model = YoloDetector(model_path=weights, preprocessing={}, conf_limit=0.25)
    paths = [ultralytics_images["bus"], ultralytics_images["zidane"]]
    probabilities, payload = model.predict(paths)

    expected = _direct_yolo_scores(weights, [load_image(path) for path in paths], conf=0.25)
    assert len(probabilities) == len(payload) == 2
    assert len(payload[0]) > 0
    for scores, expected_scores in zip(probabilities, expected):
        assert scores.tolist() == pytest.approx(expected_scores, abs=1e-4)
    for scores, boxes in zip(probabilities, payload):
        assert scores.device.type == "cpu"
        assert len(scores) == len(boxes)
        assert scores.tolist() == sorted(scores.tolist(), reverse=True)
        assert all(score >= 0.25 for score in scores.tolist())
        for box in boxes:
            assert isinstance(box, BoundingBox)
            assert 0.0 <= box.x_min <= box.x_max <= 1.0
            assert 0.0 <= box.y_min <= box.y_max <= 1.0
    _assert_ordinary_tensors(probabilities)


def test_yolo_applies_configured_preprocessing(yolo_weights, ultralytics_images):
    """Configured transforms run before YOLO: a horizontal flip config gives the same
    detections as running ultralytics on the mirrored image."""
    weights = yolo_weights("yolo26n.pt")
    model = YoloDetector(
        model_path=weights, preprocessing={"random_horizontal_flip": {"p": 1.0}}, conf_limit=0.25
    )
    probabilities, _ = model.predict([ultralytics_images["bus"]])
    mirrored = ImageOps.mirror(load_image(ultralytics_images["bus"]))
    expected = _direct_yolo_scores(weights, [mirrored], conf=0.25)
    assert probabilities[0].tolist() == pytest.approx(expected[0], abs=1e-4)


def test_yolo_missing_non_asset_path_raises_without_network(tmp_path, monkeypatch):
    """A missing file that is not an ultralytics asset name fails immediately and offline."""
    attempts = []

    def record_connect(self, address):
        attempts.append(address)
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", record_connect)
    with pytest.raises(FileNotFoundError):
        YoloDetector(model_path=tmp_path / "rodnet.pt", preprocessing={}, conf_limit=0.25)
    assert attempts == []


def test_yolo_rejects_non_pil_preprocessing_output(yolo_weights, ultralytics_images):
    """YOLO preprocesses internally, so configured transforms must keep PIL images."""
    model = YoloDetector(
        model_path=yolo_weights("yolo26n.pt"), preprocessing={"pil_to_tensor": {}}, conf_limit=0.25
    )
    with pytest.raises(TypeError):
        model.predict([ultralytics_images["bus"]])


def _imagenet_taxa(count: int) -> list[dict]:
    """Placeholder taxonomy entries, one per ImageNet class index."""
    return [{"genus": f"Genus{i}", "species": f"species{i}"} for i in range(count)]


def test_yolo_classifier_maps_probabilities_to_taxonomy(yolo_weights, ultralytics_images):
    """The pretrained ImageNet classifier: full probability vector plus sorted,
    conf_limit-filtered classifications carrying the configured taxonomy."""
    taxa = _imagenet_taxa(1000)
    model = YoloClassifier(
        model_path=yolo_weights("yolo26n-cls.pt"), preprocessing={}, conf_limit=0.05, classes=taxa
    )
    probabilities, payload = model.predict([ultralytics_images["bus"]])

    probs, classifications = probabilities[0], payload[0]
    assert probs.shape == (1000,)
    assert probs.device.type == "cpu"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    confidences = [c.confidence for c in classifications]
    assert confidences == sorted(confidences, reverse=True)
    assert len(classifications) == int((probs >= 0.05).sum())
    for classification in classifications:
        index = int(classification.species.removeprefix("species"))
        assert classification.genus == f"Genus{index}"
        assert classification.confidence == pytest.approx(float(probs[index]))
    _assert_ordinary_tensors(probabilities)


def test_yolo_classifier_taxonomy_count_mismatch_raises(yolo_weights):
    """The number of taxonomy entries must equal the model's number of classes."""
    with pytest.raises(ValueError):
        YoloClassifier(
            model_path=yolo_weights("yolo26n-cls.pt"),
            preprocessing={},
            conf_limit=0.05,
            classes=_imagenet_taxa(999),
        )


# ------------------------------------------------------------------ SpeciesNet (stub)


def make_speciesnet_stub(detections: list[dict]):
    """Create a stand-in for `speciesnet.detector.SpeciesNetDetector`.

    It mimics the SDK surface used by the pipeline: construction from a model name,
    `device` / `model` attributes (initially not on CPU, like a GPU host), and
    `preprocess(PIL image)` / `predict(filepath, preprocessed)` returning SDK-shaped
    detections (relative [x_min, y_min, width, height] boxes, sorted by confidence).
    """
    instances = []

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
            instances.append(self)

        def preprocess(self, img):
            """Accept only PIL images, as the SDK does."""
            assert isinstance(img, Image.Image)
            return img

        def predict(self, filepath, img):
            """Return the configured detections, sorted by confidence like the SDK."""
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
            st.floats(min_value=SPECIESNET_FLOOR_F32, max_value=1.0, width=32), min_size=count, max_size=count, unique=True
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
def test_speciesnet_keeps_animal_boxes_as_xyxy(colour_images, detections, conf_limit):
    """Only animal boxes >= conf_limit survive, converted from xywh to xyxy, sorted by
    score, without labels, and within the image."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("ml_pipeline.models.model.SpeciesNetDetector", make_speciesnet_stub(detections))
        model = SpeciesnetDetector(preprocessing={}, conf_limit=conf_limit)
        probabilities, payload = model.predict(colour_images[:1])

    expected = sorted(
        (d for d in detections if d["label"] == "animal" and d["conf"] >= conf_limit),
        key=lambda d: d["conf"],
        reverse=True,
    )
    scores, boxes = probabilities[0], payload[0]
    assert scores.tolist() == pytest.approx([d["conf"] for d in expected])
    assert len(boxes) == len(expected)
    for box, detection in zip(boxes, expected):
        x, y, w, h = detection["bbox"]
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == pytest.approx((x, y, x + w, y + h))
        assert -1e-9 <= box.x_min <= box.x_max <= 1.0 + 1e-9
        assert -1e-9 <= box.y_min <= box.y_max <= 1.0 + 1e-9


@pytest.fixture
def speciesnet_stub(monkeypatch):
    """Install a SpeciesNet stub with one animal and one human detection."""
    stub = make_speciesnet_stub(
        [
            _sdk_detection("animal", 0.8, [0.1, 0.2, 0.3, 0.4]),
            _sdk_detection("human", 0.9, [0.5, 0.5, 0.2, 0.2]),
        ]
    )
    monkeypatch.setattr("ml_pipeline.models.model.SpeciesNetDetector", stub)
    return stub


def test_speciesnet_defaults_to_v403b(speciesnet_stub):
    """Without model_path the pinned SpeciesNet v4.0.3b Kaggle model is requested."""
    SpeciesnetDetector(preprocessing={}, conf_limit=0.1)
    assert speciesnet_stub.instances[-1].model_name == SPECIESNET_V403B


@pytest.mark.parametrize("model_path", ["./models/speciesnet/", "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"])
def test_speciesnet_passes_model_path_unchanged(speciesnet_stub, model_path):
    """Local folders and Kaggle ids reach the SDK exactly as configured."""
    SpeciesnetDetector(model_path=model_path, preprocessing={}, conf_limit=0.1)
    assert speciesnet_stub.instances[-1].model_name == model_path


def test_speciesnet_runs_on_cpu(speciesnet_stub):
    """The SDK picks its own device (the stub pretends CUDA); the pipeline forces CPU."""
    SpeciesnetDetector(preprocessing={}, conf_limit=0.1)
    detector = speciesnet_stub.instances[-1]
    assert detector.device == "cpu"
    assert detector.model.moved
    assert all(p.device.type == "cpu" for p in detector.model.parameters())


@pytest.mark.parametrize("batch_size", [None, 1])
def test_speciesnet_one_result_per_image(speciesnet_stub, colour_images, batch_size):
    """Several images (batch_size None or 1) give one result each; humans are dropped."""
    model = SpeciesnetDetector(preprocessing={}, conf_limit=0.1, batch_size=batch_size)
    probabilities, payload = model.predict(colour_images[:3])
    assert len(probabilities) == len(payload) == 3
    for scores, boxes in zip(probabilities, payload):
        assert scores.tolist() == pytest.approx([0.8])
        assert _box_coords(boxes) == pytest.approx([0.1, 0.2, 0.4, 0.6])
    _assert_ordinary_tensors(probabilities)


def test_speciesnet_hub_download_not_implemented(speciesnet_stub, hub_stub):
    """SpeciesNet is not published on the Hugging Face Hub."""
    with pytest.raises(NotImplementedError):
        SpeciesnetDetector(hf_repo="org/speciesnet", hf_filename="model.pt", preprocessing={}, conf_limit=0.1)
    assert hub_stub == []


@pytest.mark.parametrize(
    "bad_args",
    [
        {"preprocessing": {"resize": {"size": [16, 16]}}},
        {"preprocessing": 3},  # not a mapping at all
        {"batch_size": 2},
        {"conf_limit": 0.005},
    ],
)
def test_speciesnet_invalid_settings_raise(speciesnet_stub, bad_args):
    """SpeciesNet preprocesses itself, works per image and has a 0.01 score floor."""
    kwargs = {"preprocessing": {}, "conf_limit": 0.1} | bad_args
    with pytest.raises(ValueError):
        SpeciesnetDetector(**kwargs)
    assert speciesnet_stub.instances == []


@pytest.mark.parametrize("conf_limit", ["0.5", None])
def test_speciesnet_non_numeric_conf_limit_raises(speciesnet_stub, conf_limit):
    """A non-numeric conf_limit gets the explicit TypeError, not a comparison error."""
    with pytest.raises(TypeError, match="conf_limit"):
        SpeciesnetDetector(preprocessing={}, conf_limit=conf_limit)


@pytest.mark.parametrize("preprocessing", [{}, None])
def test_speciesnet_accepts_empty_preprocessing(speciesnet_stub, preprocessing):
    """An empty preprocessing node, `{}` or YAML null, is accepted."""
    SpeciesnetDetector(preprocessing=preprocessing, conf_limit=0.1)
    assert len(speciesnet_stub.instances) == 1


# -------------------------------------------------------- SpeciesNet (real, opt-in)


@pytest.mark.speciesnet_integration
def test_real_speciesnet_detects_rodent_and_ignores_people(ultralytics_images):
    """Real SpeciesNet inference. Run with RUN_SPECIESNET_INTEGRATION=1.

    SPECIESNET_MODEL may name a local model folder or Kaggle id; otherwise the pinned
    default (v4.0.3b) is loaded from the Kaggle cache, downloading it on first use.
    """
    if os.environ.get("RUN_SPECIESNET_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPECIESNET_INTEGRATION=1 to run real SpeciesNet inference")
    model_kwargs = {}
    if "SPECIESNET_MODEL" in os.environ:
        model_kwargs["model_path"] = os.environ["SPECIESNET_MODEL"]

    model = SpeciesnetDetector(preprocessing={}, conf_limit=0.25, **model_kwargs)
    rodent = Path(__file__).parent / "data" / "rodent.jpg"
    probabilities, payload = model.predict([rodent, ultralytics_images["zidane"]])

    assert len(payload[0]) >= 1
    for box in payload[0]:
        assert 0.0 <= box.x_min <= box.x_max <= 1.0
        assert 0.0 <= box.y_min <= box.y_max <= 1.0
    # people only: no animal box with a confident score
    assert all(score < 0.5 for score in probabilities[1].tolist())
