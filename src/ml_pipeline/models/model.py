"""Model handling: build detectors and classifiers from config files and run them on one image.

Every model follows one contract:

- it is built from keyword arguments or, equivalently, from a YAML config file with the same
  keys plus a `type` key naming the model class (`load_model`);
- `info` (name, version, description) names the model in API responses;
- weights come from exactly one source: a local `model_path`, or a Hugging Face Hub file
  (`hf_repo` + `hf_filename`, optional `hf_revision`);
- a `preprocessing` node (possibly empty) holds torchvision v2 transforms;
- `predict(image)` takes one RGB PIL image (loading files is the caller's job), preprocesses it,
  runs the model and returns `(probabilities, payload)`: detectors return box scores and labelled
  `Detection`s, classifiers return the full probability vector and `Classification`s.

There is no batching: every call processes one image. All inference runs on CPU.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from speciesnet.detector import SpeciesNetDetector
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tv_functional
from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

from ml_pipeline.schemas.analysis import BoundingBox, Classification, Detection, ModelInfo, Taxon

# ------------------------------------------------------------------ module helpers


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


def _validate_conf_limit(conf_limit: Any) -> None:
    """Reject non-numeric confidence limits and values outside [0, 1] (including NaN)."""
    if isinstance(conf_limit, bool) or not isinstance(conf_limit, (int, float)):
        raise TypeError(f"conf_limit must be a number, got {conf_limit!r}")
    if not 0.0 <= conf_limit <= 1.0:
        raise ValueError(f"conf_limit must be in [0, 1], got {conf_limit}")


def _validated_info(info: Any) -> ModelInfo:
    """Validate a model's `info`; misspelled keys are errors, not silently dropped.

    The check is done here rather than with `extra="forbid"` on `ModelInfo`, so the API schema
    stays unchanged.
    """
    if isinstance(info, Mapping):
        unknown = sorted(set(info) - set(ModelInfo.model_fields), key=str)  # YAML keys may be non-strings
        if unknown:
            raise ValueError(f"unknown info keys {unknown}; allowed: {list(ModelInfo.model_fields)}")
    return ModelInfo.model_validate(info)


def _validated_label_map(label_map: Any) -> dict[str, str]:
    """Check a `label_map` node: a mapping of non-empty label strings; `None` means no renaming."""
    if label_map is None:
        return {}
    if not isinstance(label_map, Mapping):
        raise TypeError(f"label_map must be a mapping of model labels to output labels, got {label_map!r}")
    for source, target in label_map.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(f"label_map entries must map strings to strings, got {source!r}: {target!r}")
        if source == "" or target == "":
            raise ValueError(f"label_map entries must not be empty strings, got {source!r}: {target!r}")
    return dict(label_map)


def _validated_labels(labels: Any) -> tuple[str, ...]:
    """Check a torch detector's `labels`: a non-empty sequence of unique, non-empty strings."""
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise TypeError(f"labels must be a list of label strings, got {labels!r}")
    if not all(isinstance(label, str) for label in labels):
        raise TypeError(f"labels must be strings, got {labels!r}")
    if len(labels) == 0:
        raise ValueError("labels must name at least one class")
    if any(label == "" for label in labels):
        raise ValueError(f"labels must not be empty strings, got {labels!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be unique, got {labels!r}")
    return tuple(labels)


def _outside_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Elementwise test for values outside [0, 1]; NaN counts as outside."""
    return ~((values >= 0) & (values <= 1))


# --------------------------------------------------------------------- base classes


class ModelBase[T](ABC):
    """Common construction and `predict` flow of all models.

    Subclasses implement `_load_model` (backend loading) and `_infer` (one preprocessed image to
    its result). Subclasses with extra arguments validate them before calling `super().__init__`,
    so configuration errors surface before any download.

    Args:
        info: Name, version and description of the model (a `ModelInfo` or its mapping).
        preprocessing: torchvision v2 transforms, see `build_transform`. Required; may be empty.
        conf_limit: Minimum confidence in [0, 1] of returned boxes / classes.
        model_path: Local weights or backend identifier.
        hf_repo: Hugging Face Hub repository with the weights (instead of `model_path`).
        hf_filename: File in `hf_repo`.
        hf_revision: Optional revision of `hf_repo`.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: An argument has an invalid value (including an invalid `info`).
    """

    DEVICE = "cpu"  # all inference runs on CPU: the inference server has no GPU

    def __init__(
        self,
        *,
        info: ModelInfo | Mapping[str, Any],
        preprocessing: Mapping[str, Mapping[str, Any]] | None,
        conf_limit: float,
        model_path: str | Path | None = None,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        hf_revision: str | None = None,
    ):
        self.info = _validated_info(info)
        _validate_conf_limit(conf_limit)
        self.conf_limit = conf_limit
        # Build the transform before resolving weights: a config typo fails before a download.
        self.transform = build_transform(preprocessing)
        weights = resolve_weights(model_path, hf_repo, hf_filename, hf_revision)
        self.model = self._load_model(weights)

    def predict(self, image: Image.Image) -> tuple[torch.Tensor, list[T]]:
        """Run the model on one image.

        Args:
            image: An RGB PIL image.

        Returns:
            `(probabilities, payload)` with CPU tensors. Detectors: 1-D tensor of the kept box
            scores and the `Detection` list in the same (descending) order. Classifiers: the full
            probability vector and the `Classification` list of classes with probability
            >= `conf_limit`, descending.

        Raises:
            TypeError: `image` is not a PIL image, or the preprocessing produced inputs the
                backend cannot take.
            ValueError: `image` is not RGB, or the model's output breaks its contract.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"predict takes one PIL image, got {type(image).__name__}")
        if image.mode != "RGB":
            raise ValueError(f"predict takes RGB images, got mode {image.mode!r}")
        return self._infer(self._preprocess(image))

    def _preprocess(self, image: Image.Image) -> Any:
        """Apply the configured transforms to the image."""
        return self.transform(image)

    @abstractmethod
    def _load_model(self, weights: str | Path) -> Any:
        """Load the backend model from the resolved weights location."""

    @abstractmethod
    def _infer(self, preprocessed: Any) -> tuple[torch.Tensor, list[T]]:
        """Run the model on one preprocessed image; return its (CPU tensor, payload) pair."""


class DetectorBase(ModelBase[Detection]):
    """Detectors: `predict` returns box scores and labelled, not yet classified `Detection`s.

    Args:
        label_map: Optional renaming of model labels (e.g. `{"cat": "animal"}`); its keys must be
            labels the model can emit. Unmapped labels are returned verbatim.
        **kwargs: See `ModelBase`.

    Raises:
        TypeError: `label_map` is not a mapping of strings to strings.
        ValueError: `label_map` has empty strings or keys the loaded model does not emit.
    """

    def __init__(self, *, label_map: Mapping[str, str] | None = None, **kwargs):
        self.label_map = _validated_label_map(label_map)
        super().__init__(**kwargs)
        # Checked after loading, because YOLO's label set comes from the weights.
        unknown = sorted(set(self.label_map) - set(self.labels))
        if unknown:
            raise ValueError(f"label_map keys {unknown} are not labels of this model, which emits {list(self.labels)}")

    @property
    @abstractmethod
    def labels(self) -> tuple[str, ...]:
        """The labels the loaded model can emit (before `label_map`)."""

    def _make_detections(
        self, xyxy: torch.Tensor, scores: torch.Tensor, labels: Sequence[str]
    ) -> tuple[torch.Tensor, list[Detection]]:
        """Keep boxes with score >= conf_limit, sort them by descending score and label them.

        Args:
            xyxy: `[K, 4]` relative box corners on CPU.
            scores: `[K]` box scores on CPU.
            labels: K model labels, before `label_map`.

        Returns:
            The kept scores and their `Detection`s (empty `classifications`), in the same order.
        """
        kept = torch.nonzero(scores >= self.conf_limit).flatten()
        # stable sort: equal scores keep the backend's order, so results are deterministic
        kept = kept[torch.argsort(scores[kept], descending=True, stable=True)]
        detections = [
            Detection(
                detected_class=self._output_label(labels[index]),
                confidence=float(scores[index]),
                bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
            for index, (x_min, y_min, x_max, y_max) in zip(kept.tolist(), xyxy[kept].tolist())
        ]
        # indexing copies, so callers get an ordinary tensor even if the backend computed in inference mode
        return scores[kept], detections

    def _output_label(self, label: str) -> str:
        """Apply `label_map` to one model label."""
        if label in self.label_map:
            return self.label_map[label]
        else:
            return label


class ClassifierBase(ModelBase[Classification]):
    """Classifiers: `predict` returns probability vectors and taxonomy-carrying classifications.

    Args:
        classes: One taxonomy mapping (fields of `Taxon`) per model output index, in order.
        **kwargs: See `ModelBase`.

    Raises:
        TypeError: `classes` is not a list.
        ValueError: `classes` is empty or a taxonomy entry is invalid (e.g. no genus).
    """

    def __init__(self, *, classes: Sequence[Mapping[str, Any]], **kwargs):
        if isinstance(classes, (str, Mapping)) or not isinstance(classes, Sequence):
            raise TypeError(f"classes must be a list of taxonomy entries, got {classes!r}")
        if len(classes) == 0:
            raise ValueError("classes must have one taxonomy entry per model output")
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

        If the file does not exist and its basename is an asset name, ultralytics downloads the
        pretrained weights to that path, so a mistyped `models/custom/yolo26n.pt` gets the stock
        weights: that is ultralytics' own behaviour.

        Raises:
            FileNotFoundError: The path does not exist and is no ultralytics asset name.
            ValueError: The weights are for another task (e.g. classifier weights for a detector).
        """
        path = Path(weights)
        # Without this check ultralytics would query the GitHub API (30 s timeout) before failing.
        if not path.exists() and path.name not in GITHUB_ASSETS_NAMES:
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        model = YOLO(str(weights), task=self.task)
        # ultralytics takes the task from the checkpoint, ignoring `task=`; a mismatch would only
        # fail later, on every predict call
        if model.task != self.task:
            raise ValueError(f"{weights} is a YOLO '{model.task}' model; {type(self).__name__} needs '{self.task}'")
        return model

    def _run_yolo(self, preprocessed: Any, **predict_kwargs) -> Any:
        """Run ultralytics on one PIL image (it letterboxes and normalises itself); return its result.

        Raises:
            TypeError: The configured preprocessing did not keep a PIL image.
        """
        if not isinstance(preprocessed, Image.Image):
            raise TypeError(
                f"YOLO models need PIL images; preprocessing produced {type(preprocessed).__name__} "
                "(remove tensor-producing transforms such as pil_to_tensor)"
            )
        return self.model.predict(source=preprocessed, device=self.DEVICE, verbose=False, **predict_kwargs)[0]


class YoloDetector(_UltralyticsMixin, DetectorBase):
    """Ultralytics YOLO detector. All boxes are kept and labelled with the model's class names.

    Boxes are relative to the preprocessed image, so they match the original image only if
    preprocessing does not crop, pad or flip. `model_path` may be an ultralytics asset name such
    as `yolo26n.pt` (see `_UltralyticsMixin._load_model`). Ultralytics keeps at most 300 boxes
    per image (`max_det` default) and drops boxes whose score equals `conf_limit` exactly (it
    filters with `>`). See `DetectorBase` / `ModelBase` for the arguments.
    """

    task = "detect"

    @property
    def labels(self) -> tuple[str, ...]:
        """The class names stored in the YOLO weights."""
        return tuple(self.model.names.values())

    def _infer(self, preprocessed: Any) -> tuple[torch.Tensor, list[Detection]]:
        """Detect and label boxes in one PIL image."""
        boxes = self._run_yolo(preprocessed, conf=self.conf_limit).boxes
        labels = [self.model.names[int(class_index)] for class_index in boxes.cls.tolist()]
        return self._make_detections(boxes.xyxyn.cpu(), boxes.conf.cpu(), labels)


class YoloClassifier(_UltralyticsMixin, ClassifierBase):
    """Ultralytics YOLO classifier.

    `model_path` may be an ultralytics asset name such as `yolo26n-cls.pt` (see
    `_UltralyticsMixin._load_model`). See `ClassifierBase` / `ModelBase` for the arguments.

    Raises:
        ValueError: The number of `classes` differs from the model's number of classes.
    """

    task = "classify"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.classes) != len(self.model.names):
            raise ValueError(f"{len(self.classes)} taxonomy entries for a model with {len(self.model.names)} classes")

    def _infer(self, preprocessed: Any) -> tuple[torch.Tensor, list[Classification]]:
        """Classify one PIL image."""
        # clone: ultralytics computes under inference mode; callers get an ordinary tensor
        probabilities = self._run_yolo(preprocessed).probs.data.cpu().float().clone()
        return probabilities, self._make_classifications(probabilities)


# -------------------------------------------------------------------------- torch


class _TorchModelMixin:
    """Shared loading and calling of plain `.pt` torch modules.

    The file format and its trust requirements are documented on `TorchClassifier` and
    `TorchDetector`.

    Model input: a float32 `[1, 3, H, W]` CPU tensor; the image is converted with
    `ToImage + ToDtype(float32, scale=True)` (values in [0, 1]) and then passed through the
    configured transforms. The leading batch dimension keeps modules ready for later batching.
    The module is in eval mode and runs under `torch.inference_mode()`.
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

    def _forward(self, preprocessed: Any) -> Any:
        """Run the module on one preprocessed image, as a batch of one.

        Raises:
            TypeError: The configured preprocessing did not produce a tensor.
        """
        if not isinstance(preprocessed, torch.Tensor):
            raise TypeError("torch models need tensors; preprocessing must not convert them back (e.g. to_pil_image)")
        # as_subclass strips torchvision's tv_tensors.Image wrapper: the module gets a plain tensor
        batch = preprocessed.as_subclass(torch.Tensor).unsqueeze(0)
        with torch.inference_mode():
            return self.model(batch)

    @staticmethod
    def _check_shape(output: Any, rank: int, last_dim: int, expected: str) -> None:
        """Raise ValueError naming the `expected` output unless `output` is one tensor of that shape.

        Args:
            output: What the module returned.
            rank: Required number of dimensions; the first one must be the batch of one.
            last_dim: Required size of the last dimension.
            expected: Human-readable description of the contract, for the error message.
        """
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"model must return {expected}, got {type(output).__name__}")
        if output.ndim != rank or output.shape[0] != 1 or output.shape[-1] != last_dim:
            raise ValueError(f"model must return {expected}, got shape {tuple(output.shape)}")


class TorchClassifier(_TorchModelMixin, ClassifierBase):
    """Classifier from a pickled torch module that returns a single `[1, C]` tensor.

    Column i of the output corresponds to `classes[i]`.

    The `.pt` file must hold a whole pickled module (`torch.save(model)`), loaded with
    `weights_only=False`: unpickling can run arbitrary code, so load trusted files only, and
    the package defining the module class must be importable here.

    Args:
        apply_softmax: True if the module returns logits, False if it returns probabilities
            (values in [0, 1] summing to 1).
        **kwargs: See `ClassifierBase` / `ModelBase`.

    Raises:
        TypeError: `apply_softmax` is not a bool.
    """

    def __init__(self, *, apply_softmax: bool, **kwargs):
        if not isinstance(apply_softmax, bool):
            raise TypeError(f"apply_softmax must be true or false, got {apply_softmax!r}")
        self.apply_softmax = apply_softmax
        super().__init__(**kwargs)

    def _infer(self, preprocessed: torch.Tensor) -> tuple[torch.Tensor, list[Classification]]:
        """Classify one preprocessed tensor."""
        output = self._forward(preprocessed)
        self._check_output(output)
        if self.apply_softmax:
            probabilities = torch.softmax(output.float(), dim=1)
        else:
            probabilities = output.float()
        # clone: the output was computed under inference mode, and without softmax `probabilities`
        # is still that inference tensor; callers get an ordinary tensor
        row = probabilities[0].cpu().clone()
        return row, self._make_classifications(row)

    def _check_output(self, output: Any) -> None:
        """Raise ValueError naming the expected output if the module broke the contract."""
        expected = f"a single tensor [1, C={len(self.classes)}]"
        self._check_shape(output, rank=2, last_dim=len(self.classes), expected=expected)
        if self.apply_softmax:
            if not torch.isfinite(output).all():
                raise ValueError("classifier logits (apply_softmax: true) must be finite")
        else:
            if _outside_unit_interval(output).any():
                raise ValueError("classifier probabilities (apply_softmax: false) must lie in [0, 1]")
            if not torch.allclose(output.float().sum(dim=1), torch.ones(1), atol=1e-4):
                raise ValueError("classifier probabilities (apply_softmax: false) must sum to 1")


class TorchDetector(_TorchModelMixin, DetectorBase):
    """Detector from a pickled torch module that returns a single `[1, K, 6]` tensor.

    Each row is `(x1, y1, x2, y2, score, class_index)`, with corners relative (0-1) to the model
    input tensor; they are passed through unchanged, so they are relative to the original image
    only if preprocessing does not crop, pad or flip. `class_index` is an integer value indexing
    `labels`. K may vary between calls. Rows with score 0 are padding and are always dropped.

    The `.pt` file must hold a whole pickled module (`torch.save(model)`), loaded with
    `weights_only=False`: unpickling can run arbitrary code, so load trusted files only, and
    the package defining the module class must be importable here.

    Args:
        labels: The label of each class index, in order.
        **kwargs: See `DetectorBase` / `ModelBase`.

    Raises:
        TypeError: `labels` is not a list of strings.
        ValueError: `labels` is empty or has empty or duplicate entries.
    """

    def __init__(self, *, labels: Sequence[str], **kwargs):
        self._labels = _validated_labels(labels)
        super().__init__(**kwargs)

    @property
    def labels(self) -> tuple[str, ...]:
        """The configured labels, indexed by the output's class column."""
        return self._labels

    def _infer(self, preprocessed: torch.Tensor) -> tuple[torch.Tensor, list[Detection]]:
        """Detect and label boxes in one preprocessed tensor."""
        output = self._forward(preprocessed)
        self._check_output(output)
        rows = output[0].float().cpu()
        # Mask computed once: padding (score 0) and rows below conf_limit are dropped here, and
        # only kept rows must carry valid boxes and class indices.
        kept = rows[(rows[:, 4] > 0) & (rows[:, 4] >= self.conf_limit)]
        self._check_kept_rows(kept)
        labels = [self._labels[int(class_index)] for class_index in kept[:, 5].tolist()]
        return self._make_detections(kept[:, :4], kept[:, 4], labels)

    def _check_output(self, output: Any) -> None:
        """Raise ValueError naming the expected output if the module broke the contract."""
        expected = "a single tensor [1, K, 6] of (x1, y1, x2, y2, score, class_index)"
        self._check_shape(output, rank=3, last_dim=6, expected=expected)
        if _outside_unit_interval(output[..., 4]).any():
            raise ValueError("detector scores must lie in [0, 1]")

    def _check_kept_rows(self, rows: torch.Tensor) -> None:
        """Raise ValueError for kept rows with invalid boxes or class indices."""
        xyxy, class_index = rows[:, :4], rows[:, 5]
        if _outside_unit_interval(xyxy).any():
            raise ValueError("detector box corners must lie in [0, 1] (relative coordinates)")
        if (xyxy[:, 0] > xyxy[:, 2]).any() or (xyxy[:, 1] > xyxy[:, 3]).any():
            raise ValueError("detector boxes must have x1 <= x2 and y1 <= y2")
        # NaN fails the integer test, so it is rejected, too
        invalid = (class_index != torch.round(class_index)) | (class_index < 0) | (class_index >= len(self._labels))
        if invalid.any():
            raise ValueError(f"detector class indices must be integers in [0, {len(self._labels)})")


# --------------------------------------------------------------------- SpeciesNet


class SpeciesnetDetector(DetectorBase):
    """SpeciesNet's detector component (MegaDetector), returning all boxes with their SDK label.

    Labels are `animal`, `human` and `vehicle`. SpeciesNet preprocesses images itself
    (letterbox to 1280), so `preprocessing` must be present but empty. The SDK drops boxes
    below 0.01, so `conf_limit` must be >= 0.01.

    Kaggle models are downloaded and cached by the SDK (`kagglehub`). If the model's `info.json`
    lists the detector weights as a URL (as v4.0.3b does: MegaDetector v5a on GitHub), the SDK
    also downloads that file with `requests` on first use. Importing the SDK makes Pillow load
    truncated images process-wide (`PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`).

    Args:
        info: See `ModelBase`.
        model_path: Local SpeciesNet model folder (with `info.json`) or a `kaggle:` id; Kaggle
            models are downloaded and cached by the SDK. Defaults to SpeciesNet v4.0.3b.
        preprocessing: Must be empty (`{}` or `None`).
        conf_limit: Minimum box score, in [0.01, 1].
        hf_repo: Not supported: SpeciesNet is not published on the Hugging Face Hub.
        hf_filename: Only valid with `hf_repo`.
        hf_revision: Only valid with `hf_repo`.
        label_map: See `DetectorBase`; keys must be in `LABELS`.

    Raises:
        NotImplementedError: `hf_repo` was given.
        TypeError: `preprocessing` is not a mapping, or `conf_limit` is not a number.
        ValueError: Non-empty preprocessing, or `conf_limit` < 0.01.
    """

    DEFAULT_MODEL_PATH = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
    LABELS = ("animal", "human", "vehicle")  # speciesnet.constants.Detection
    MIN_CONF_LIMIT = 0.01  # the SDK's fixed DETECTION_THRESHOLD
    # The SDK's predict only echoes its filepath into the result; models never see paths.
    SDK_FILEPATH_PLACEHOLDER = "<in-memory image>"

    def __init__(
        self,
        *,
        info: ModelInfo | Mapping[str, Any],
        preprocessing: Mapping[str, Mapping[str, Any]] | None,
        conf_limit: float,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        hf_revision: str | None = None,
        label_map: Mapping[str, str] | None = None,
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
        _validate_conf_limit(conf_limit)  # type check first, so the comparison below is safe
        if conf_limit < self.MIN_CONF_LIMIT:
            raise ValueError(f"SpeciesNet drops boxes below {self.MIN_CONF_LIMIT}; conf_limit must be >= that")
        super().__init__(
            info=info,
            preprocessing=preprocessing,
            conf_limit=conf_limit,
            model_path=model_path,
            hf_filename=hf_filename,
            hf_revision=hf_revision,
            label_map=label_map,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """SpeciesNet's fixed detector labels."""
        return self.LABELS

    def _load_model(self, weights: str | Path) -> SpeciesNetDetector:
        """Load the SDK detector and force it onto the CPU."""
        detector = SpeciesNetDetector(str(weights))
        # The SDK picks cuda/mps/cpu itself and has no device argument; `predict` reads these
        # instance attributes, so setting them is how inference is kept on the CPU. (On a GPU
        # host the checkpoint is briefly loaded onto the GPU before this move.)
        detector.device = self.DEVICE
        detector.model.to(self.DEVICE)
        return detector

    def _infer(self, preprocessed: Image.Image) -> tuple[torch.Tensor, list[Detection]]:
        """Run the SDK on one image and convert its boxes to relative xyxy corners."""
        # `predict` only accepts RGB PIL images, so `preprocess` never returns None here and the
        # SDK's "failures" result (which has no "detections") cannot occur.
        prediction = self.model.predict(self.SDK_FILEPATH_PLACEHOLDER, self.model.preprocess(preprocessed))
        detections = prediction["detections"]
        xyxy = torch.tensor(
            [[x, y, x + width, y + height] for x, y, width, height in (d["bbox"] for d in detections)],
            dtype=torch.float32,
        ).reshape(-1, 4)
        scores = torch.tensor([d["conf"] for d in detections], dtype=torch.float32)
        return self._make_detections(xyxy, scores, [d["label"] for d in detections])


# ---------------------------------------------------------------------- config files


def load_model(path: str | Path) -> ModelBase:
    """Build a model from a YAML config file.

    The required `type` key names the model class; all other keys are its constructor arguments,
    so missing required keys and unknown keys raise.

    Args:
        path: The config file.

    Returns:
        The model.

    Raises:
        FileNotFoundError: The config file does not exist.
        yaml.YAMLError: The file is not valid YAML.
        TypeError: The file's top level is not a mapping, or it has missing/unknown keys.
        ValueError: `type` is missing or names no model class, or an argument is invalid.
    """
    with open(path) as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping):
        raise TypeError(f"config {path} must contain a mapping at the top level")
    kwargs = dict(config)
    model_class = _model_class(kwargs.pop("type", None))
    return model_class(**kwargs)


def _model_class(model_type: Any) -> type[ModelBase]:
    """Map a config's `type` value to its model class."""
    match model_type:
        case "YoloDetector":
            return YoloDetector
        case "SpeciesnetDetector":
            return SpeciesnetDetector
        case "TorchDetector":
            return TorchDetector
        case "YoloClassifier":
            return YoloClassifier
        case "TorchClassifier":
            return TorchClassifier
        case _:
            raise ValueError(
                "config key `type` must be one of YoloDetector, SpeciesnetDetector, TorchDetector, "
                f"YoloClassifier, TorchClassifier; got {model_type!r}"
            )
