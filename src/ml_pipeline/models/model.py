"""Model handling: load detectors and classifiers and run them on lists of image paths.

Every model follows one contract (see planning/plan_model_handling.md):

- it is built from keyword arguments or, equivalently, from a YAML config file with the
  same keys (`from_config`);
- weights come from exactly one source: a local `model_path`, or a Hugging Face Hub file
  (`hf_repo` + `hf_filename`, optional `hf_revision`);
- a `preprocessing` node (possibly empty) holds torchvision v2 transforms;
- `predict(image_paths)` loads the images, preprocesses them, runs the model in batches and
  returns `(probabilities, payload)` with one entry per image, in input order:
  detectors return box scores and `BoundingBox` lists (class-agnostic), classifiers return
  full probability vectors and `Classification` lists.

All inference runs on CPU.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from speciesnet.detector import SpeciesNetDetector
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tv_functional
from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

from ml_pipeline.schemas.analysis import BoundingBox, Classification, Taxon

# ------------------------------------------------------------------ module helpers


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB, with its EXIF orientation applied.

    Args:
        path: Path of the image file.

    Returns:
        The decoded RGB image.

    Raises:
        FileNotFoundError: The file does not exist.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def build_transform(preprocessing: Mapping[str, Mapping[str, Any]] | None) -> v2.Transform:
    """Build a torchvision v2 pipeline from a `preprocessing` config node.

    The node maps transform names to keyword arguments, in application order, e.g.
    `{"resize": {"size": [448, 448], "antialias": True}}`. Names are matched case- and
    underscore-insensitively against the `v2.Transform` subclasses, so `random_horizontal_flip`
    resolves to `RandomHorizontalFlip` and `pil_to_tensor` to `PILToTensor`. Transforms whose
    arguments YAML cannot express (`ToDtype`, `Compose`, `RandomApply`, `Lambda`, ...) are not
    supported.

    Args:
        preprocessing: The node; an empty mapping or `None` (a bare YAML key) means no transform.

    Returns:
        The composed transform, or `v2.Identity()` for an empty node.

    Raises:
        TypeError: The node or a transform's kwargs are not mappings, or a name is not a string.
        ValueError: A name does not match a concrete v2 transform.
    """
    if preprocessing is None:
        return v2.Identity()
    if not isinstance(preprocessing, Mapping):
        raise TypeError(f"preprocessing must be a mapping of transform names to kwargs, got {preprocessing!r}")

    available = _available_transforms()
    transforms = []
    for name, kwargs in preprocessing.items():
        if not isinstance(name, str):
            raise TypeError(f"transform names in preprocessing must be strings, got {name!r}")
        transform_class = available.get(_normalise_name(name))
        if transform_class is None:
            raise ValueError(f"unknown torchvision v2 transform in preprocessing: {name!r}")
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"kwargs of transform {name!r} must be a mapping, got {kwargs!r}")
        transforms.append(transform_class(**kwargs))

    if not transforms:
        # v2.Compose([]) raises, so an empty node needs its own explicit identity
        return v2.Identity()
    return v2.Compose(transforms)


def _available_transforms() -> dict[str, type[v2.Transform]]:
    """Map normalised names to the concrete `v2.Transform` subclasses of torchvision."""
    return {
        _normalise_name(name): obj
        for name, obj in vars(v2).items()
        # the abstract base class itself is not a usable transform
        if isinstance(obj, type) and issubclass(obj, v2.Transform) and obj is not v2.Transform
    }


def _normalise_name(name: str) -> str:
    """Lower-case a transform name and drop underscores, for name matching."""
    return name.replace("_", "").lower()


def resolve_weights(
    model_path: str | Path | None,
    hf_repo: str | None,
    hf_filename: str | None,
    hf_revision: str | None,
) -> str | Path:
    """Return the location of the model weights, downloading them from the HF Hub if configured.

    A local `model_path` is returned untouched: it is not resolved, normalised or checked for
    existence, so backend-specific identifiers (YOLO asset names, SpeciesNet `kaggle:` ids)
    keep their meaning. Each backend's loader reports missing files.

    Args:
        model_path: Local weights (or a backend identifier).
        hf_repo: Hugging Face Hub repository id.
        hf_filename: File inside `hf_repo`; required with `hf_repo`.
        hf_revision: Optional branch, tag or commit of `hf_repo`.

    Returns:
        `model_path` unchanged, or the local path of the downloaded Hub file.

    Raises:
        ValueError: Not exactly one weights source, or incomplete/stray Hub arguments.
    """
    if model_path is not None and hf_repo is not None:
        raise ValueError("give either model_path or hf_repo, not both")
    elif model_path is None and hf_repo is None:
        raise ValueError("no weights source: give model_path or hf_repo (+ hf_filename)")
    elif model_path is not None:
        if hf_filename is not None or hf_revision is not None:
            raise ValueError("hf_filename / hf_revision are only valid together with hf_repo")
        return model_path
    else:
        if hf_filename is None:
            raise ValueError("hf_repo needs hf_filename: the file to download from the repository")
        return Path(hf_hub_download(repo_id=hf_repo, filename=hf_filename, revision=hf_revision))


def ensure_pil(image: Any) -> Image.Image:
    """Check that preprocessing kept a PIL image, for backends that preprocess PIL images themselves.

    Args:
        image: One preprocessed image.

    Returns:
        `image`, unchanged.

    Raises:
        TypeError: The configured preprocessing turned the image into something else.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"this model needs PIL images; preprocessing produced {type(image).__name__} "
            "(remove tensor-producing transforms such as pil_to_tensor)"
        )
    return image


# --------------------------------------------------------------------- base classes


class ModelBase[T](ABC):
    """Common construction and `predict` flow of all models.

    Subclasses implement `_load_model` (backend loading) and `_infer` (one batch of
    preprocessed images to per-image results). Subclasses with extra arguments validate them
    before calling `super().__init__`, so configuration errors surface before any download.

    Args:
        preprocessing: torchvision v2 transforms, see `build_transform`. Required; may be empty.
        conf_limit: Minimum confidence in [0, 1] of returned boxes / classes.
        batch_size: Images per model call; `None` passes all images of a `predict` call at once.
        model_path: Local weights or backend identifier.
        hf_repo: Hugging Face Hub repository with the weights (instead of `model_path`).
        hf_filename: File in `hf_repo`.
        hf_revision: Optional revision of `hf_repo`.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has an invalid value.
    """

    DEVICE = "cpu"  # all inference runs on CPU: the inference server has no GPU

    def __init__(
        self,
        *,
        preprocessing: Mapping[str, Mapping[str, Any]] | None,
        conf_limit: float,
        batch_size: int | None = None,
        model_path: str | Path | None = None,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        hf_revision: str | None = None,
    ):
        _validate_conf_limit(conf_limit)
        _validate_batch_size(batch_size)
        self.conf_limit = conf_limit
        self.batch_size = batch_size
        # Build the transform before resolving weights: a config typo fails before a download.
        self.transform = build_transform(preprocessing)
        weights = resolve_weights(model_path, hf_repo, hf_filename, hf_revision)
        self.model = self._load_model(weights)

    @classmethod
    def from_config(cls, path: str | Path) -> Self:
        """Build a model from a YAML file whose top-level keys are the constructor arguments.

        Args:
            path: The config file.

        Returns:
            The model.

        Raises:
            TypeError: The file's top level is not a mapping, or it has missing/unknown keys.
        """
        with open(path) as file:
            config = yaml.safe_load(file)
        if not isinstance(config, Mapping):
            raise TypeError(f"config {path} must contain a mapping at the top level")
        return cls(**config)

    def predict(self, image_paths: list[str | Path]) -> tuple[list[torch.Tensor], list[list[T]]]:
        """Run the model on images.

        Args:
            image_paths: Image files.

        Returns:
            `(probabilities, payload)`, one entry per image in input order, all tensors on CPU.
            Detectors: 1-D tensor of kept box scores and the `BoundingBox` list in the same
            (descending) order. Classifiers: the full probability vector and the
            `Classification` list of classes with probability >= `conf_limit`, descending.

        Raises:
            TypeError: `image_paths` is a single path instead of a list, or the preprocessing
                produced inputs the backend cannot take.
            FileNotFoundError: An image does not exist.
            ValueError: The model's output breaks its contract, or images cannot be batched.
        """
        if isinstance(image_paths, (str, Path)):
            raise TypeError("predict takes a list of image paths, not a single path")
        if len(image_paths) == 0:
            return [], []
        chunk_size = len(image_paths) if self.batch_size is None else self.batch_size
        results: list[tuple[torch.Tensor, list[T]]] = []
        for start in range(0, len(image_paths), chunk_size):
            chunk = image_paths[start : start + chunk_size]
            inputs = [self._preprocess(load_image(path)) for path in chunk]
            results.extend(self._infer(inputs, [str(path) for path in chunk]))
        return [tensor for tensor, _ in results], [items for _, items in results]

    def _preprocess(self, image: Image.Image) -> Any:
        """Apply the configured transforms to one loaded image."""
        return self.transform(image)

    @abstractmethod
    def _load_model(self, weights: str | Path) -> Any:
        """Load the backend model from the resolved weights location."""

    @abstractmethod
    def _infer(self, inputs: list[Any], paths: list[str]) -> list[tuple[torch.Tensor, list[T]]]:
        """Run one batch of preprocessed images; return one (CPU tensor, payload) pair per image."""


def _validate_conf_limit(conf_limit: Any) -> None:
    """Reject non-numeric confidence limits and values outside [0, 1] (including NaN)."""
    if isinstance(conf_limit, bool) or not isinstance(conf_limit, (int, float)):
        raise TypeError(f"conf_limit must be a number, got {conf_limit!r}")
    if not 0.0 <= conf_limit <= 1.0:
        raise ValueError(f"conf_limit must be in [0, 1], got {conf_limit}")


def _validate_batch_size(batch_size: Any) -> None:
    """Accept None or an int >= 1; bool is rejected explicitly because `True >= 1` holds."""
    if batch_size is None:
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int or None, got {batch_size!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")


class DetectorBase(ModelBase[BoundingBox]):
    """Class-agnostic detectors: `predict` returns box scores and `BoundingBox` lists."""

    def _make_boxes(self, xyxy: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, list[BoundingBox]]:
        """Keep boxes with score >= conf_limit and sort them by descending score.

        Args:
            xyxy: `[K, 4]` relative box corners on CPU.
            scores: `[K]` box scores on CPU.

        Returns:
            The kept scores and their `BoundingBox`es, in the same order.
        """
        keep = scores >= self.conf_limit
        xyxy, scores = xyxy[keep], scores[keep]
        # stable sort: equal scores keep the backend's order, so results are deterministic
        order = torch.argsort(scores, descending=True, stable=True)
        xyxy, scores = xyxy[order], scores[order]
        boxes = [
            BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            for x_min, y_min, x_max, y_max in xyxy.tolist()
        ]
        return scores, boxes


class ClassifierBase(ModelBase[Classification]):
    """Classifiers: `predict` returns probability vectors and taxonomy-carrying classifications.

    Args:
        classes: One taxonomy mapping (fields of `Taxon`) per model output index, in order.
        **kwargs: See `ModelBase`.

    Raises:
        ValueError: A taxonomy entry is invalid (e.g. no genus).
    """

    def __init__(self, *, classes: Sequence[Mapping[str, Any]], **kwargs):
        self.classes = [Taxon.model_validate(entry) for entry in classes]
        self._taxon_fields = [taxon.model_dump() for taxon in self.classes]
        super().__init__(**kwargs)

    def _make_classifications(self, probabilities: torch.Tensor) -> list[Classification]:
        """Turn one `[C]` probability vector into classifications >= conf_limit, descending."""
        kept = torch.nonzero(probabilities >= self.conf_limit).flatten()
        # stable sort: equal probabilities keep class-index order, so results are deterministic
        kept = kept[torch.argsort(probabilities[kept], descending=True, stable=True)]
        return [
            Classification(**self._taxon_fields[index], confidence=confidence)
            for index, confidence in zip(kept.tolist(), probabilities[kept].tolist())
        ]


# --------------------------------------------------------------------------- YOLO


class _UltralyticsMixin:
    """Shared loading and calling of ultralytics YOLO models (`task` set by the subclass)."""

    task: str

    def _load_model(self, weights: str | Path) -> YOLO:
        """Load a YOLO model; known asset names (e.g. `yolo26n.pt`) are downloaded by ultralytics.

        A missing path whose basename *is* an asset name (e.g. a mistyped
        `models/custom/yolo26n.pt`) downloads the stock pretrained weights to that path:
        that is ultralytics' own behaviour.

        Raises:
            FileNotFoundError: The path does not exist and is no ultralytics asset name.
        """
        path = Path(weights)
        # Without this check ultralytics would query the GitHub API (30 s timeout) before failing.
        if not path.exists() and path.name not in GITHUB_ASSETS_NAMES:
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        return YOLO(str(weights), task=self.task)

    def _run_yolo(self, inputs: list[Any], **predict_kwargs) -> list[Any]:
        """Run ultralytics on one batch of PIL images (it letterboxes and normalises itself)."""
        images = [ensure_pil(image) for image in inputs]
        return self.model.predict(source=images, device=self.DEVICE, verbose=False, **predict_kwargs)


class YoloDetector(_UltralyticsMixin, DetectorBase):
    """Ultralytics YOLO detector. All boxes are kept regardless of their class; labels are dropped.

    `model_path` may be an ultralytics asset name such as `yolo26n.pt`: if the file does not
    exist, ultralytics downloads the pretrained weights to that path. A mistyped path ending in
    an asset name therefore downloads the stock weights. Ultralytics keeps at most 300 boxes
    per image (`max_det` default) and drops boxes whose score equals `conf_limit` exactly (it
    filters with `>`). See `ModelBase` for the arguments.
    """

    task = "detect"

    def _infer(self, inputs: list[Any], paths: list[str]) -> list[tuple[torch.Tensor, list[BoundingBox]]]:
        """Detect boxes in one batch of PIL images."""
        results = self._run_yolo(inputs, conf=self.conf_limit)
        return [self._make_boxes(r.boxes.xyxyn.cpu(), r.boxes.conf.cpu()) for r in results]


class YoloClassifier(_UltralyticsMixin, ClassifierBase):
    """Ultralytics YOLO classifier.

    `model_path` may be an ultralytics asset name such as `yolo26n-cls.pt`, which ultralytics
    downloads if the file does not exist (also for a mistyped path ending in such a name).
    See `ClassifierBase` / `ModelBase` for the arguments.

    Raises:
        ValueError: The number of `classes` differs from the model's number of classes.
    """

    task = "classify"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.classes) != len(self.model.names):
            raise ValueError(
                f"{len(self.classes)} taxonomy entries for a model with {len(self.model.names)} classes"
            )

    def _infer(self, inputs: list[Any], paths: list[str]) -> list[tuple[torch.Tensor, list[Classification]]]:
        """Classify one batch of PIL images."""
        results = self._run_yolo(inputs)
        # clone: ultralytics computes under inference mode; callers get ordinary tensors
        probabilities = [r.probs.data.cpu().float().clone() for r in results]
        return [(probs, self._make_classifications(probs)) for probs in probabilities]


# -------------------------------------------------------------------------- torch


class _TorchModelMixin:
    """Shared loading and calling of plain `.pt` torch modules (see the contract in the plan, 3.4).

    The file format and its trust requirements are documented on `TorchClassifier` and
    `TorchDetector`.

    Model input: a float32 `[B, 3, H, W]` CPU tensor; each image is converted with
    `ToImage + ToDtype(float32, scale=True)` (values in [0, 1]) and then passed through the
    configured transforms, so all images must end up the same size (configure a resize).
    `1 <= B <= batch_size`, or all images of the `predict` call without `batch_size`. The
    module is in eval mode and runs under `torch.inference_mode()`.
    """

    def _load_model(self, weights: str | Path) -> torch.nn.Module:
        """Unpickle the module onto the CPU and put it into eval mode.

        Raises:
            ImportError: The package defining the pickled module class is not importable.
            TypeError: The file does not contain an `nn.Module`.
        """
        # weights_only=False is required for whole pickled modules (the torch default is True).
        try:
            module = torch.load(weights, map_location=self.DEVICE, weights_only=False)
        except (ModuleNotFoundError, AttributeError) as error:
            # pickle stores the module class by import path; anticipated when its package is missing
            raise ImportError(
                f"{weights} holds a module whose class is not importable here ({error}); "
                "install the package that defines it"
            ) from error
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{weights} does not contain a torch.nn.Module but {type(module).__name__}")
        return module.eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert to a float [0, 1] tensor, then apply the configured transforms."""
        tensor = tv_functional.to_dtype(tv_functional.to_image(image), torch.float32, scale=True)
        return self.transform(tensor)

    def _forward(self, inputs: list[torch.Tensor]) -> Any:
        """Stack one batch and run the module on it.

        Raises:
            TypeError: The configured preprocessing did not produce tensors.
            ValueError: The preprocessed images differ in size.
        """
        if not all(isinstance(tensor, torch.Tensor) for tensor in inputs):
            raise TypeError("torch models need tensors; preprocessing must not convert them back (e.g. to_pil_image)")
        shapes = {tuple(tensor.shape) for tensor in inputs}
        if len(shapes) != 1:
            raise ValueError(
                f"preprocessed images differ in size {sorted(shapes)}; add a resize to the preprocessing node"
            )
        # as_subclass strips torchvision's tv_tensors.Image wrapper: the module gets plain tensors
        batch = torch.stack([tensor.as_subclass(torch.Tensor) for tensor in inputs])
        with torch.inference_mode():
            return self.model(batch)

    @staticmethod
    def _check_shape(output: Any, batch_size: int, rank: int, last_dim: int, expected: str) -> None:
        """Raise ValueError naming the `expected` output unless `output` is one tensor of that shape.

        Args:
            output: What the module returned.
            batch_size: B of the input batch; the output's first dimension.
            rank: Required number of dimensions.
            last_dim: Required size of the last dimension.
            expected: Human-readable description of the contract, for the error message.
        """
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"model must return {expected}, got {type(output).__name__}")
        if output.ndim != rank or output.shape[0] != batch_size or output.shape[-1] != last_dim:
            raise ValueError(f"model must return {expected}, got shape {tuple(output.shape)}")


def _outside_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Elementwise test for values outside [0, 1]; NaN counts as outside."""
    return ~((values >= 0) & (values <= 1))


class TorchClassifier(_TorchModelMixin, ClassifierBase):
    """Classifier from a pickled torch module that returns a single `[B, C]` tensor.

    Column i of the output corresponds to `classes[i]`.

    The `.pt` file must hold a whole pickled module (`torch.save(model)`), loaded with
    `weights_only=False`: unpickling can run arbitrary code, so load trusted files only, and
    the package defining the module class must be importable here.

    Args:
        apply_softmax: True if the module returns logits, False if it returns probabilities
            (rows in [0, 1] summing to 1).
        **kwargs: See `ClassifierBase` / `ModelBase`.

    Raises:
        TypeError: `apply_softmax` is not a bool.
    """

    def __init__(self, *, apply_softmax: bool, **kwargs):
        if not isinstance(apply_softmax, bool):
            raise TypeError(f"apply_softmax must be true or false, got {apply_softmax!r}")
        self.apply_softmax = apply_softmax
        super().__init__(**kwargs)

    def _infer(self, inputs: list[torch.Tensor], paths: list[str]) -> list[tuple[torch.Tensor, list[Classification]]]:
        """Classify one batch of preprocessed tensors."""
        output = self._forward(inputs)
        self._check_output(output, len(inputs))
        if self.apply_softmax:
            probabilities = torch.softmax(output.float(), dim=1)
        else:
            probabilities = output.float()
        # clone: the output was computed under inference mode; callers get ordinary tensors
        rows = probabilities.cpu().clone()
        return [(row, self._make_classifications(row)) for row in rows]

    def _check_output(self, output: Any, batch_size: int) -> None:
        """Raise ValueError naming the expected output if the module broke the contract."""
        expected = f"a single tensor [B={batch_size}, C={len(self.classes)}]"
        self._check_shape(output, batch_size, rank=2, last_dim=len(self.classes), expected=expected)
        if self.apply_softmax:
            if not torch.isfinite(output).all():
                raise ValueError("classifier logits (apply_softmax: true) must be finite")
        else:
            if _outside_unit_interval(output).any():
                raise ValueError("classifier probabilities (apply_softmax: false) must lie in [0, 1]")
            if not torch.allclose(output.float().sum(dim=1), torch.ones(batch_size), atol=1e-4):
                raise ValueError("classifier probabilities (apply_softmax: false) must sum to 1 per image")


class TorchDetector(_TorchModelMixin, DetectorBase):
    """Detector from a pickled torch module that returns a single `[B, K, 5]` tensor.

    Each row is `(x1, y1, x2, y2, score)`, with corners relative (0-1) to the model input
    tensor; they are passed through unchanged, so they are relative to the original image
    only if preprocessing does not crop, pad or flip. K may vary between calls. Rows with
    score 0 are padding and are always dropped. See `ModelBase` for the arguments.

    The `.pt` file must hold a whole pickled module (`torch.save(model)`), loaded with
    `weights_only=False`: unpickling can run arbitrary code, so load trusted files only, and
    the package defining the module class must be importable here.
    """

    def _infer(self, inputs: list[torch.Tensor], paths: list[str]) -> list[tuple[torch.Tensor, list[BoundingBox]]]:
        """Detect boxes in one batch of preprocessed tensors."""
        output = self._forward(inputs)
        self._check_output(output, len(inputs))
        output = output.float().cpu()
        # Mask computed once: padding (score 0) and rows below conf_limit are dropped here,
        # and only kept rows must carry valid boxes.
        kept = (output[..., 4] > 0) & (output[..., 4] >= self.conf_limit)
        self._check_kept_boxes(output[kept][:, :4])
        return [self._make_boxes(rows[mask, :4], rows[mask, 4]) for rows, mask in zip(output, kept)]

    def _check_output(self, output: Any, batch_size: int) -> None:
        """Raise ValueError naming the expected output if the module broke the contract."""
        expected = f"a single tensor [B={batch_size}, K, 5] of (x1, y1, x2, y2, score)"
        self._check_shape(output, batch_size, rank=3, last_dim=5, expected=expected)
        if _outside_unit_interval(output[..., 4]).any():
            raise ValueError("detector scores must lie in [0, 1]")

    @staticmethod
    def _check_kept_boxes(xyxy: torch.Tensor) -> None:
        """Raise ValueError for kept boxes outside [0, 1] or with inverted corners."""
        if _outside_unit_interval(xyxy).any():
            raise ValueError("detector box corners must lie in [0, 1] (relative coordinates)")
        if (xyxy[:, 0] > xyxy[:, 2]).any() or (xyxy[:, 1] > xyxy[:, 3]).any():
            raise ValueError("detector boxes must have x1 <= x2 and y1 <= y2")


# --------------------------------------------------------------------- SpeciesNet


class SpeciesnetDetector(DetectorBase):
    """SpeciesNet's detector component (MegaDetector), returning only its `animal` boxes, unlabelled.

    SpeciesNet preprocesses images itself (letterbox to 1280), so `preprocessing` must be
    present but empty. It processes one image per SDK call, so `batch_size` may only be
    `None` or 1. The SDK drops boxes below 0.01, so `conf_limit` must be >= 0.01.

    Kaggle models are downloaded and cached by the SDK (`kagglehub`). If the model's `info.json`
    lists the detector weights as a URL (as v4.0.3b does: MegaDetector v5a on GitHub), the SDK
    also downloads that file with `requests` on first use.

    Args:
        model_path: Local SpeciesNet model folder (with `info.json`) or a `kaggle:` id; Kaggle
            models are downloaded and cached by the SDK. Defaults to SpeciesNet v4.0.3b.
        preprocessing: Must be empty (`{}` or `None`).
        conf_limit: Minimum box score, in [0.01, 1].
        batch_size: `None` or 1.
        hf_repo: Not supported: SpeciesNet is not published on the Hugging Face Hub.
        hf_filename: Only valid with `hf_repo`.
        hf_revision: Only valid with `hf_repo`.

    Raises:
        NotImplementedError: `hf_repo` was given.
        TypeError: `preprocessing` is not a mapping, or `conf_limit` is not a number.
        ValueError: Non-empty preprocessing, `batch_size` other than None/1, or `conf_limit` < 0.01.
    """

    DEFAULT_MODEL_PATH = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
    KEPT_LABEL = "animal"
    MIN_CONF_LIMIT = 0.01  # the SDK's fixed DETECTION_THRESHOLD

    def __init__(
        self,
        *,
        preprocessing: Mapping[str, Mapping[str, Any]] | None,
        conf_limit: float,
        batch_size: int | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        hf_revision: str | None = None,
    ):
        if hf_repo is not None:
            raise NotImplementedError(
                "SpeciesNet is published on Kaggle, not on the Hugging Face Hub; "
                "use model_path (a local model folder or a kaggle: id) instead of hf_repo"
            )
        if preprocessing is not None and not isinstance(preprocessing, Mapping):
            raise TypeError(f"preprocessing must be a mapping of transform names to kwargs, got {preprocessing!r}")
        if preprocessing is not None and len(preprocessing) != 0:
            raise ValueError("SpeciesNet preprocesses images itself; its preprocessing node must be empty")
        if batch_size is not None and batch_size != 1:
            raise ValueError("SpeciesNet processes one image per call; batch_size must be omitted or 1")
        _validate_conf_limit(conf_limit)  # type check first, so the comparison below is safe
        if conf_limit < self.MIN_CONF_LIMIT:
            raise ValueError(f"SpeciesNet drops boxes below {self.MIN_CONF_LIMIT}; conf_limit must be >= that")
        super().__init__(
            preprocessing=preprocessing,
            conf_limit=conf_limit,
            batch_size=batch_size,
            model_path=model_path,
            hf_filename=hf_filename,
            hf_revision=hf_revision,
        )

    def _load_model(self, weights: str | Path) -> SpeciesNetDetector:
        """Load the SDK detector and force it onto the CPU."""
        detector = SpeciesNetDetector(str(weights))
        # The SDK picks cuda/mps/cpu itself and has no device argument; `predict` reads these
        # instance attributes, so setting them is how inference is kept on the CPU. (On a GPU
        # host the checkpoint is briefly loaded onto the GPU before this move.)
        detector.device = self.DEVICE
        detector.model.to(self.DEVICE)
        return detector

    def _infer(self, inputs: list[Image.Image], paths: list[str]) -> list[tuple[torch.Tensor, list[BoundingBox]]]:
        """Detect boxes image by image (the SDK has no batch API for its detector component)."""
        return [self._detect(image, path) for image, path in zip(inputs, paths)]

    def _detect(self, image: Image.Image, path: str) -> tuple[torch.Tensor, list[BoundingBox]]:
        """Run the SDK on one image and keep its animal boxes as relative xyxy corners."""
        # load_image always yields an image, so `preprocess` never returns None here and the
        # SDK's "failures" result (which has no "detections") cannot occur.
        prediction = self.model.predict(path, self.model.preprocess(image))
        animals = [d for d in prediction["detections"] if d["label"] == self.KEPT_LABEL]
        xyxy = torch.tensor(
            [[x, y, x + width, y + height] for x, y, width, height in (d["bbox"] for d in animals)],
            dtype=torch.float32,
        ).reshape(-1, 4)
        scores = torch.tensor([d["conf"] for d in animals], dtype=torch.float32)
        return self._make_boxes(xyxy, scores)
