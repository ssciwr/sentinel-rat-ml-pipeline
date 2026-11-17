# Plan: extensible model handling (`src/ml_pipeline/models/model.py`)

Stage: PLANNING (revision 8). No code is changed until this plan is approved.

## 1. Requirements

From the request:

R1. Every model's `predict` takes a list of image paths, loads the images, runs preprocessing,
    and returns a tuple `(probabilities, payload)`. Detectors return bounding boxes as the payload, and
    classifiers return classes. Batching depends on the model.
R2. Every model can be built from a config file holding the model path. Loading uses the backend's own
    method (YOLO, torch, SpeciesNet).
R3. Models are downloaded with `huggingface_hub` when the config or the `__init__` args contain the HF args.
R4. Every classifier config holds the full taxonomy of each species class (see `Classification` schema).
R5. Every config must have a `preprocessing` node. The node is either empty or holds torchvision
    transforms in the format of the `transforms` node in `smartrodent_experiments/configs/train_supcon_config.yaml`.
R6. Follow `smartrodent_experiments/CLAUDE.md` (SOLID, flat control flow, explicit branches, no silent
    defaults, Google docstrings, TDD, Hypothesis property tests, no module-level variables).
R7. Keep it simple and extensible. Do not restrict future models with the same behaviour and input surface.

From the user's follow-up decisions:

R8. Torch models are **plain `.pt` files that `torch.load(..., weights_only=False)` restores to an `nn.Module`**. Pickle is accepted
    for now. `.pt2` / `torch.export` was dropped as too complex.
R9. SpeciesNet does its own preprocessing. We do not apply transforms to it.
R10. SpeciesNet is not published on the HF Hub. HF args for SpeciesNet raise `NotImplementedError` with an explanation.
R11. **Inference always runs on CPU.** There is no `device` key; `"cpu"` is hard-coded for torch, ultralytics and SpeciesNet.
R12. **Detectors are class-agnostic.** They return boxes and scores only, with no class labels. The crops are classified
     downstream by a classifier. Off-the-shelf detector labels are never returned (SpeciesNet uses its label only to filter, R13).
R13. SpeciesNet keeps only boxes it labels `animal` (it drops `human` and `vehicle`) and returns them without the label.
     YOLO detectors keep all boxes regardless of their class (COCO or custom species classes) and drop the label.
R14. `speciesnet` is a regular runtime dependency (not optional). As in `smartrodent_experiments`, the SpeciesNet model is
     downloaded from Kaggle by the SDK if it is not available locally. **The default model is always SpeciesNet `v4.0.3b`**
     (`kaggle:google/speciesnet/pyTorch/v4.0.3b/1`), not the SDK's `DEFAULT_MODEL` (`v4.0.3a`).
R15. `batch_size` is an **optional** key. If given, it is honoured. If absent, all images of one `predict` call are passed to
     the model as a single batch.
R16. `apply_softmax` (TorchClassifier) is kept: most of our models return raw logits.

## 2. Findings that drive the design

### 2.1 Plain `.pt` torch models
- `torch.save(model, path)` pickles the whole `nn.Module`. `torch.load(path, map_location="cpu", weights_only=False)` restores it.
  Pickle stores the class only by import path (`pkg.module.ClassName`), so **the package that defines the model class
  must be importable in this pipeline's environment**. Loading unpickles arbitrary objects, so only trusted files may be
  loaded, which is accepted per R8. We document this in the docstrings. A consequence: serving a model pickled from a class in
  another project makes that package a runtime dependency of the pipeline.
- `weights_only=False` is required for whole-module pickles. The torch default is `True` since 2.6, so we pass the
  argument explicitly instead of relying on a default.
- After loading we call `model.eval()`. The model runs eagerly, so any batch size works if the module supports it,
  and `map_location="cpu"` places all tensors on CPU (no separate `.to()` call). The `.pt2` caveats (fixed shapes, in-graph device moves) no longer apply.
- Considered and rejected:
  - TorchScript `.pt` (`torch.jit`): torch 2.15 warns that it is "not supported in Python 3.14+ and may break", and this venv runs Python 3.14.
  - state_dict plus a model class named in config: avoids pickle, but needs extra config. Not needed while pickle is accepted, and it can be added later as a separate loader.

### 2.2 SpeciesNet (from `smartrodent_experiments/src/smartrodent/yolo_dataset_creation.py` and the SDK)
- The SDK's detector component `speciesnet.detector.SpeciesNetDetector(model_name)` exposes
  `preprocess(PIL.Image) -> PreprocessedImage` (letterbox to 1280) and
  `predict(filepath, preprocessed) -> {"detections": [{"label", "conf", "bbox": [x_min, y_min, w, h] (relative)}]}`.
  We load the image ourselves with the shared loader and hand it to SpeciesNet's own `preprocess`. **One image per call**: there is no batching.
- `model_name` is either a local folder holding `info.json` or a Kaggle identifier (`kaggle:...`). For a Kaggle identifier the SDK
  downloads the model with `kagglehub` (cache: `~/.cache/kagglehub`, reused on later runs). `smartrodent_experiments` uses the SDK's
  `DEFAULT_MODEL` (`v4.0.3a`); this pipeline deliberately defaults to `v4.0.3b` (R14). `SpeciesnetDetector` makes `model_path` optional, with the explicit, documented default
  `kaggle:google/speciesnet/pyTorch/v4.0.3b/1` (R14), stored as the class attribute `DEFAULT_MODEL_PATH` (no module-level variables). It passes `model_path`
  to the SDK unchanged. `v4.0.3b` (489 MB, which includes the detector weights) is already in the local Kaggle cache, and the
  id resolves offline (the reviewer verified the kagglehub cache lookup). On a fresh machine, the first construction downloads the
  Kaggle bundle (~489 MB) plus the MegaDetector weights (~281 MB, listed as a GitHub URL in `info.json`).
  Its `info.json` has `"type": "full_image"` and the detector `md_v5a.0.1.pt` (MegaDetector v5a). The SDK's own `DEFAULT_MODEL` is `v4.0.3a`.
  The `a`/`b` suffix marks the SDK's two classifier modes (`a`: `always_crop`, `b`: `full_image`, per the `info.json` type).
  Only the detector component is used here.
- `predict` has a fixed floor of 0.01 (`DETECTION_THRESHOLD`). `SpeciesnetDetector` therefore raises a `ValueError` for `conf_limit < 0.01`
  instead of silently ignoring the lower limit.
- SpeciesNet selects its device itself (`cuda` / `mps` / `cpu`) and has no device argument. To enforce CPU (R11), after construction we
  set `detector.device = "cpu"` and call `detector.model.to("cpu")`. `predict` reads `self.device`. This relies on SDK instance
  attributes, so we document it in a comment and cover it with a test.
- The SDK's `preprocess` returns `None` (and `predict` then reports a `failures` entry instead of `detections`) only for a `None` image.
  `load_image` always returns an image or raises, so that path cannot occur and needs no handling.
- If a SpeciesNet `info.json` lists the detector as a URL, the SDK downloads it with `requests`. We document this as SDK behaviour.
- Google publishes SpeciesNet on Kaggle and GitHub. The only HF copy is a third-party mirror, hence R10.

### 2.3 YOLO (ultralytics)
- `ultralytics.YOLO(path, task="detect" | "classify")`, then `.predict(source=list[PIL.Image], conf=..., device="cpu", verbose=False)`.
  YOLO letterboxes and normalises internally and processes a list of images in one call (batched).
- Detection results: `r.boxes.xyxyn`, `r.boxes.conf` (`r.boxes.cls` is ignored, R12). Classification results: `r.probs.data` (C,), `model.names`.
  These tensors are moved to CPU.
- The YOLO detector keeps at most `max_det=300` boxes by default (ultralytics' cap). We document this and do not configure it.
- Pretrained default names: for a `model_path` that does not exist locally and is an ultralytics asset name (`yolo26n.pt`,
  `yolo26n-cls.pt`, ...), `YOLO(...)` downloads the pretrained weights from the ultralytics GitHub release to exactly that path
  (`utils/downloads.py::attempt_download_asset`; it first checks the ultralytics `weights_dir`). This is YOLO's own loading
  behaviour (R2), so the YOLO classes pass `model_path` through to `YOLO()` unchanged. We never build random models from `.yaml`.
- Edge cases, handled explicitly in the YOLO `_load_model`:
  - a path that does not exist and whose basename is **not** an asset name (e.g. a typo `models/rodnet.pt`): ultralytics would
    query the GitHub API (30 s timeout) before failing. We pre-check and raise `FileNotFoundError` immediately when
    `not path.exists() and path.name not in ultralytics.utils.downloads.GITHUB_ASSETS_NAMES`;
  - a missing path whose basename **is** an asset name (e.g. a mistyped `models/custom/yolo26n.pt`) downloads the stock pretrained
    weights. This is ultralytics' intended behaviour; it is documented in the YOLO class docstrings.

### 2.4 Transforms node format (`train_supcon_config.yaml` + `scripts/train_supcon.py::_build_transform`)
- The node is a mapping from a snake_case transform name to a kwargs mapping. The experiments node also has `image_size`, which is not a transform.
- Checked: v2 transforms accept string interpolation (`"bilinear"`) and keep PIL in → PIL out. `v2.Compose([])` raises `ValueError`.

## 3. Design

All of this lives in `src/ml_pipeline/models/model.py`, as the existing stubs do.

```
ModelBase[T] (ABC)                         # T = BoundingBox | Classification
 ├── __init__(*, preprocessing, conf_limit, batch_size=None,
 │            model_path=None, hf_repo=None, hf_filename=None, hf_revision=None)
 │      batch_size=None means one batch with all images of a predict call (R15); otherwise it must be an int >= 1
 │      (bool and float are rejected explicitly, since `True >= 1` would pass).
 │      order: (1) validate scalar args, (2) self.transform = build_transform(preprocessing),
 │             (3) path = resolve_weights(...), (4) self.model = self._load_model(path).
 │      A typo in config therefore fails before any download. Subclasses validate their own args
 │      (classes/Taxon, apply_softmax, SpeciesNet rules) BEFORE calling super().__init__().
 │      Checks that need the loaded model (YOLO len(classes) == len(model.names)) run after it.
 ├── from_config(cls, path) -> Self        # classmethod: yaml.safe_load -> cls(**cfg)
 │      top level not a mapping -> TypeError; model_path is passed on as given (relative = relative to the CWD, as everywhere in Python),
 │      so YOLO default names like `yolo26n.pt` keep their ultralytics meaning. Whether the file exists is checked by the backend loader
 │      (torch.load / YOLO / SpeciesNet raise for missing files; YOLO downloads known asset names).
 ├── predict(image_paths) -> tuple[list[Tensor], list[list[T]]]   # concrete template method
 │      load RGB images -> self._preprocess(img) each -> chunk by batch_size (or one chunk if None) -> self._infer(chunk) -> concat
 ├── _preprocess(image)  (concrete: returns self.transform(image), unchecked)
 ├── _load_model(path)   (abstract)
 └── _infer(inputs, paths) -> tuple[list[Tensor], list[list[T]]]  (abstract; returns CPU tensors)
 │
 ├── DetectorBase(ModelBase[BoundingBox])   # helper: _make_boxes(xyxyn, scores) -> filters by conf_limit, sorts desc
 │     ├── YoloDetector                      # all boxes, class ignored
 │     ├── SpeciesnetDetector                # keeps only label == KEPT_LABEL ("animal") boxes; class attribute, since SpeciesNet's label set is fixed
 │     │                                     # model_path defaults to the class attribute DEFAULT_MODEL_PATH = kaggle:google/speciesnet/pyTorch/v4.0.3b/1 (R14); passed to the SDK unchanged
 │     │                                     # before super().__init__(): hf_repo given -> NotImplementedError,
 │     │                                     # non-empty preprocessing -> ValueError, batch_size not in (None, 1) or conf_limit < 0.01 -> ValueError
 │     │                                     # (it always processes one image per SDK call)
 │     └── TorchDetector(_TorchModelMixin, DetectorBase)
 └── ClassifierBase(ModelBase[Classification])   # + required `classes: list[taxon mapping]`
       │                                         # helper: _make_classifications(probs) -> sorted, >= conf_limit
       │                                         # (width vs len(classes): YOLO checks it at load, torch in _check_output)
       ├── YoloClassifier                    (validates len(classes) == len(model.names) at load)
       └── TorchClassifier(_TorchModelMixin, ClassifierBase)  (+ required `apply_softmax: bool`)

_TorchModelMixin: _load_model = torch.load(path, map_location="cpu", weights_only=False), then .eval();
                  TypeError if the unpickled object is not an nn.Module;
                  _preprocess = ToImage + ToDtype(float32, scale=True) + config transforms -> Tensor;
                  stacks a batch (a clear ValueError if the shapes differ: preprocessing must resize);
                  runs under torch.inference_mode().
TorchClassifier / TorchDetector._infer: output = module(batch) -> self._check_output(output, len(batch)) -> convert to payload
                  (call -> check -> convert; each check is one explicit `if ... raise ValueError`, see 3.4)
                  TorchDetector computes the kept-row mask (score > 0 & score >= conf_limit) once. It validates the masked rows,
                  drops padding, and passes the rest to DetectorBase._make_boxes. The padding rule stays torch-only;
                  _make_boxes keeps only the generic >= conf_limit filter and sort.
_UltralyticsMixin: shared YOLO loading (asset-name pre-check) and calling (ensure_pil, device="cpu") for
                  YoloDetector / YoloClassifier (added in the implementation stage, analogous to _TorchModelMixin).
ensure_pil(image) -> PIL.Image   # module-level helper called by the YOLO _infer; TypeError if a transform produced a
                                 # non-PIL output (YOLO preprocesses internally)
build_transform(preprocessing) -> v2.Transform   # module-level pure function
resolve_weights(model_path, hf_repo, hf_filename, hf_revision) -> str | Path   # module-level function; HF -> hf_hub_download (Path);
                                 # a local model_path is returned untouched (not resolved, normalised or existence-checked),
                                 # so YOLO asset names and SpeciesNet `kaggle:` ids survive. SpeciesNet passes str(model_path) to the SDK.
load_image(path) -> PIL.Image (EXIF-transposed, RGB)                    # module-level function
```

Why this shape:
- **Template method in `ModelBase.predict`.** Loading, preprocessing and batching are written once. A new backend implements
  only `_load_model` and `_infer` (open/closed), and every subclass keeps the same `predict` contract (Liskov).
- **`DetectorBase` / `ClassifierBase`** turn raw model outputs into the payload types and own the config each type needs (taxonomy for classifiers, R4).
- **The torch-specific behaviour is in a mixin**, so `TorchDetector` and `TorchClassifier` share `.pt` loading and
  tensor preprocessing without a third inheritance level.
- **The PIL-output check stays out of the base `_preprocess`.** Only YOLO needs a PIL image, so it calls `ensure_pil` itself.
  SpeciesNet's transform is always the identity (R9), so its input is the loaded PIL image.
- **No separate config class.** `from_config` passes the YAML mapping straight to `__init__` as keyword args. Python raises
  an error for missing required keys (including `preprocessing`, R5) and for unknown keys, with no silent defaults. The `__init__` args and the config keys are the same surface (R2, R3).
- **Removed from the stub:** `from_hub` (the HF args on `__init__`/the config cover it), `_load_species_config` / `_load_model_config`
  (`from_config` covers them), and the unused `TypeVarTuple`.

### 3.1 `predict` contract (R1)
`predict(image_paths: list[str | Path]) -> tuple[list[torch.Tensor], list[list[T]]]`, with one entry per input image, in input order. All tensors are on CPU.
- Detectors: `probabilities[i]` is a 1-D tensor of the confidences of the kept boxes, and `payload[i]` is a
  `list[BoundingBox]` (relative xyxy) in the same order, so `probabilities[i][j]` is the score of `payload[i][j]`. Both are
  sorted by confidence (descending), and only boxes with `confidence >= conf_limit` are kept. No class labels (R12).
- Classifiers: `probabilities[i]` is the full 1-D probability vector over all C classes.
  `payload[i]` is a `list[Classification]` for every class with `p >= conf_limit`, sorted descending, with taxonomy fields filled from config.
- Empty `image_paths` returns `([], [])`. A missing file raises `FileNotFoundError` (it is not skipped).

### 3.2 Config file (YAML, R2/R3/R4/R5)
```yaml
# weights: exactly one of model_path | hf_repo (else ValueError). Exception: SpeciesNet defaults model_path, so "neither" is valid there
model_path: models/rodent-cls.pt   # SpeciesNet: a local folder with info.json or a `kaggle:` id; optional, default kaggle:google/speciesnet/pyTorch/v4.0.3b/1
# hf_repo: org/repo                # not supported for SpeciesNet (NotImplementedError)
# hf_filename: weights/best.pt     # required when hf_repo is given (ValueError otherwise) -> hf_hub_download
# hf_revision: main                # optional, passed through
conf_limit: 0.25                   # [0, 1]; SpeciesNet: [0.01, 1]
# batch_size: 16                   # optional, >= 1; absent -> all images of a predict call in one batch; SpeciesNet: absent or 1
preprocessing: {}                  # required key; `{}` or a bare `preprocessing:` (YAML null) both mean "empty" -> v2.Identity()
                                   # SpeciesNet: must be present and empty (it preprocesses itself), ValueError otherwise.
                                   # This rule is documented in the SpeciesnetDetector class docstring and in the config example.
# preprocessing:
#   resize: {size: [448, 448], interpolation: bilinear, antialias: true}
classes:                           # classifiers only: one entry per model output index, in order
  - {kingdom: Animalia, phylum: Chordata, class_name: Mammalia, order: Rodentia,
     family: Muridae, genus: Rattus, species: norvegicus, common_name: brown rat}
# apply_softmax: true              # TorchClassifier only: true if the module outputs logits
```
Preprocessing rules:
- Mapping order is application order.
- A key is matched case- and underscore-insensitively against the **`v2.Transform` subclasses** in `torchvision.transforms.v2`,
  so `random_adjust_sharpness` → `RandomAdjustSharpness` and `pil_to_tensor` → `PILToTensor`. An unknown name, or a non-transform
  attribute such as `functional`, raises `ValueError`.
- A value must be a mapping of kwargs (`{}` for none). Anything else raises `TypeError`.
- Transforms whose arguments YAML cannot express (`ToDtype`, `Compose`, `RandomApply`, `Lambda`) are unsupported, and the docstring says so.
- **Difference from the experiments format:** there is no `image_size` key. `size` goes into the transform's own kwargs.
- For YOLO, the transform output must remain a PIL image (`TypeError` otherwise). Torch models always get
  `ToImage + ToDtype(float32, scale=True)` in front of the configured transforms.

### 3.3 Taxonomy (R4)
Add a `Taxon` pydantic model to `schemas/analysis.py` with the taxonomy fields `Classification` has now, and make
`Classification(Taxon)` add `confidence`. The serialised JSON stays identical, and the existing tests must keep passing.
Classifier config entries are validated as `Taxon` at construction (before loading). The number of entries is checked against `model.names` at load (YOLO), and against the output width in `_check_output` on every call (torch).

### 3.4 Model contract for torch `.pt` models
The contract defines only what the pipeline passes to the loaded module and what the module must return. How a model is
built, trained or saved to meet it is up to the model developer.

**Input (from the pipeline).**
- A float32 tensor `[B, 3, H, W]` on CPU. Each loaded RGB image goes through `ToImage + ToDtype(float32, scale=True)`
  (values in [0, 1]), then through the configured `preprocessing` transforms in order. The final value range is whatever
  those transforms produce (for example after `normalize`).
- `H` and `W` are whatever preprocessing produces. All images in a batch must end up the same size.
- `1 <= B <= batch_size` when `batch_size` is set (the last chunk may be smaller). Without `batch_size`, B is the number of images in the `predict` call.
- The module is in eval mode and is called once per batch as `module(batch)`, under `torch.inference_mode()`. There is no autograd.

**Required output.**
- `TorchClassifier`: a single tensor `[B, C]`, where C == `len(classes)` and column i corresponds to `classes[i]`.
  The values are logits (`apply_softmax: true`) or probabilities (`apply_softmax: false`; each row in [0, 1] and summing to 1).
- `TorchDetector`: a single tensor `[B, K, 5]`, where K may vary between calls. Each row is `(x1, y1, x2, y2, score)`:
  - xyxy are relative (0–1) to the **model input tensor**. The pipeline passes them through unchanged, so they are relative to the
    original image only if preprocessing does not crop, pad or flip.
  - `score` is in [0, 1].
  - Rows with `score == 0` are **padding** and are always dropped, whatever `conf_limit` is. Of the remaining rows, those with `score >= conf_limit` are kept.

**Contract violations** raise a `ValueError` that names the expected output. The checks are grouped in one
`_check_output(output, batch_size)` per torch class; the detector's kept-row box checks live in `_check_kept_boxes`, because they
need the kept-row mask (see the design block):
- the output is not a single tensor;
- the rank or last dimension is wrong (classifier: rank 2 with C == `len(classes)`; detector: rank 3 with last dim 5);
- `output.shape[0] != B`;
- classifier with `apply_softmax: true`: non-finite logits (NaN/inf);
- classifier with `apply_softmax: false`: values outside [0, 1], or rows not summing to 1 (`torch.allclose(..., atol=1e-4)`);
- detector: a `score` outside [0, 1];
- detector, kept rows only: xyxy outside [0, 1], or `x1 > x2` or `y1 > y2`.

Range checks are written as `~((x >= 0) & (x <= 1))`, so NaN also counts as a violation and is never silently dropped.

### 3.5 Dependencies (`pyproject.toml`)
- Runtime deps: `torch`, `torchvision`, `ultralytics` (already added by the user before this task), plus `huggingface_hub` and `pyyaml`
  (already imported, currently only transitive).
- `[tool.uv] constraint-dependencies = ["opencv-python-headless>=5.0,<5.1"]`: `opencv-python` (via ultralytics) and
  `opencv-python-headless` (via speciesnet → yolov5 → roboflow) install the same `cv2` package, so their versions must match, and headless
  5.1.0.95 has no linux wheels. `required-environments` for linux x86_64 makes `uv lock` fail early if CI/server lose wheels.
- Add `speciesnet` as a **regular runtime dependency** (R14). It brings `kagglehub` (the Kaggle download), `yolov5` and a
  `setuptools<82` pin. It is imported at module level like the other backends.
- torch/torchvision come from the CPU-only PyTorch index (`[tool.uv.sources]` + an explicit `[[tool.uv.index]]`
  `https://download.pytorch.org/whl/cpu`), decided by the user (R11). This avoids ~45 `nvidia-*` CUDA packages from PyPI's linux
  wheels. `nvidia-ml-py` (a small pure-Python NVML binding required by ultralytics) remains.
- Add dev dep `hypothesis`.

## 4. Test plan (next stage, TDD; new file `tests/test_model.py`)
Tests check behaviour at the public surface (`predict`, `from_config`, `build_transform`, `resolve_weights`), using real artifacts:
tiny pickled torch modules and pretrained YOLO weights (downloaded once into the pytest cache). SpeciesNet uses an SDK stub, plus an
opt-in real-model test that CI skips.
- **Torch, with real `.pt` files made in session-scoped fixtures (`tmp_path_factory`).** Tiny `nn.Module` classes are defined at module level in
  `tests/tiny_models.py`, because pickle needs the class importable (verified under this repo's pytest setup). They are saved with
  `torch.save(model)`: a classifier with a `Dropout` layer that outputs logits, and a detector that returns fixed `[B, K, 5]` rows.
  Hypothesis tests use `settings(deadline=None, max_examples=<modest>)`. Models are built inside `@given` only where a constructor
  argument (`batch_size`, `conf_limit`) is drawn; single-image reference predictions come from a session fixture.
  Drawn limits and scores are float32 (`width=32`), because the pipeline compares in float32 tensors.
  Extra tiny modules: `ColourDetector` (one image-dependent box per image, so the per-image split of detector rows is tested)
  and `BatchSizeProbe` (one-hot at index B-1, so the observed chunk sizes prove that `batch_size` is honoured, R15).
  - `.eval()` takes effect: repeated `predict` calls on the Dropout classifier return identical probabilities.
  - Hypothesis property: `predict` returns one entry per path, in order, for any number of paths and any `batch_size` (including `None`),
    including a short last chunk. The results match with and without `batch_size` (`torch.allclose(atol=1e-6)` for probabilities,
    approximate comparison for boxes), because float32 results can differ slightly between batch shapes.
  - Classifier payload, as a Hypothesis property over `conf_limit`: sorted descending, every confidence ≥ `conf_limit`, taxonomy matches the
    output index, and probability vectors sum to 1 when softmax is applied.
  - Detector payload: `BoundingBox` lists with relative xyxy, aligned with the score tensors, confidence ≥ `conf_limit`, sorted descending.
  - Each of these raises `ValueError`: a taxonomy length mismatch, and images of different sizes without a resize.
  - A pickled non-module object raises `TypeError`.
  - Each contract violation from 3.4 raises `ValueError`. The tiny models return wrong outputs on purpose:
    - a tuple instead of a tensor
    - the wrong rank or last dimension
    - a first dimension != B
    - with `apply_softmax: false`, rows that are not probabilities
    - a detector score outside [0, 1]
    - a kept row with xyxy outside [0, 1]
    - a classifier width != `len(classes)`, checked in `_check_output` (the only place for the torch width check)
  - Padding: a detector that emits zero-score padding rows (arbitrary xyxy), run with `conf_limit: 0.0`, returns only the real rows and raises nothing.
  - More violation cases: a kept row with inverted xyxy (`x1 > x2`), and a NaN score or probability.
  - K varies: a tiny detector that returns a different K per call gives a correct payload for each call.
  - Returned tensors are on CPU.
- **Contract behaviour from 3.1.**
  - Empty `image_paths` returns `([], [])`.
  - A missing file raises `FileNotFoundError`.
  - `load_image` applies the EXIF orientation: an image with an EXIF rotate tag comes back with swapped dimensions.
  - `load_image` converts to RGB: a grayscale or RGBA input comes back as `mode == "RGB"`.
- **`build_transform`, tested as behaviour.**
  - Empty mapping and `None` behave as identity on an image.
  - Hypothesis property: for subsets and orderings of a few deterministic transforms with snake_case keys, the output on a generated
    image equals the output of the same v2 transforms composed by hand.
  - An unknown name raises `ValueError`, and so does a non-transform v2 attribute such as `functional`.
  - A non-mapping kwargs value raises `TypeError`.
- **Config and `__init__` validation.** Each of these raises an error:
  - missing `preprocessing`
  - both `model_path` and `hf_repo`; neither of them (YOLO and torch only; SpeciesNet has a default)
  - `hf_repo` without `hf_filename`
  - `conf_limit` outside [0, 1]
  - `batch_size < 1` (`ValueError`), or `batch_size` of type bool or float (`TypeError`)
  - `conf_limit` NaN (`ValueError`, like any value outside [0, 1])
  - added in the implementation stage, one test per explicit guard: a non-mapping preprocessing node, a non-numeric
    `conf_limit`, a non-bool `apply_softmax` (all `TypeError`), and `hf_filename`/`hf_revision` without `hf_repo` (`ValueError`)
  - added after the implementation review:
    - returned tensors are ordinary (not inference-mode) CPU tensors for every model type;
    - misspelled taxonomy keys are rejected (`Taxon` forbids extra keys);
    - NaN logits are rejected;
    - the abstract `transform` name is rejected, and non-string transform names raise `TypeError`;
    - `predict` with a single path raises `TypeError`;
    - torch preprocessing that leaves no tensors raises `TypeError`;
    - SpeciesNet with a non-numeric `conf_limit` raises `TypeError`;
    - an unimportable pickled class raises `ImportError`
  - unknown keys
  - a non-PIL transform output for YOLO
  - a config file whose top level is not a mapping
  - a bad preprocessing entry and a bad classifier `classes` entry, both of which must raise **before** any weights are resolved
    (checked by asserting that the HF stub was not called)

  A `from_config` round trip gives the same `predict` output as the equivalent kwargs.
- **HF.** `hf_hub_download` is replaced via monkeypatch with a stub that returns a local path. The tests check the forwarded repo,
  filename and revision, and that the model loads from the returned path. No network access.
- **YOLO, with the pretrained default models.** The tests use `yolo26n.pt` (COCO detector) and `yolo26n-cls.pt` (ImageNet classifier),
  which ultralytics downloads on first use. **This needs network access once.** A session-scoped fixture points `model_path` into the
  pytest cache (`pytestconfig.cache.mkdir("yolo_weights") / "yolo26n.pt"`), so the weights are downloaded once, reused across runs,
  and never land in the repo. In CI, `.pytest_cache` is not kept between runs, so each CI run downloads the two small files
  (`yolo26n.pt` and `yolo26n-cls.pt`, a few MB each) from the ultralytics GitHub release. The CI workflow is not changed in this task. Test images are the ones bundled with ultralytics (`ultralytics/assets/bus.jpg`, `zidane.jpg`), so the
  pretrained detector finds real boxes. The tests check:
  - payload invariants: lengths, ordering, `conf_limit`, relative boxes (all COCO classes kept, no labels), CPU tensors
  - that the payload on `bus.jpg` is non-empty at `conf_limit: 0.25` (no exact box counts are asserted)
  - that a missing non-asset path raises `FileNotFoundError` without network access
  - if the weights are not cached and the download fails (no network, `--cache-clear`, or `-p no:cacheprovider`), the fixture
    **fails** with an explicit message ("pretrained YOLO weights not cached and download failed: <reason>"). It does not skip,
    so a silent loss of YOLO coverage is impossible.
  - the classifier taxonomy-count check against `model.names` (1000 ImageNet classes; the test builds the 1000 `classes` entries
    programmatically) and a mismatch raising `ValueError`
- **SpeciesNet unit tests, with an SDK stub** (as in `smartrodent_experiments/tests/speciesnet_test_support.py`), so CI never
  downloads the ~500 MB Kaggle model. `ml_pipeline.models.model.SpeciesNetDetector` is replaced via `monkeypatch.setattr` with a stub
  whose `preprocess`/`predict` outputs follow the SDK format, plus a tiny `nn.Module` as `.model`. The tests check:
  - xywh→xyxy conversion, as a Hypothesis property over SDK-valid inputs (`x_min + w <= 1`, `y_min + h <= 1`; the SDK clips boxes):
    boxes stay in [0, 1] with x_min ≤ x_max and y_min ≤ y_max
  - only `animal` boxes are returned (`human`/`vehicle` dropped), with no label; `conf_limit` filtering, order, and alignment with the scores
  - when `model_path` is omitted, the stub receives `kaggle:google/speciesnet/pyTorch/v4.0.3b/1`; a given `model_path` is passed through unchanged
  - after construction the detector's `device` is `"cpu"` and its model parameters are on CPU (a state check)
  - HF args raise `NotImplementedError`
  - non-empty preprocessing, a `batch_size` other than `None`/1, and `conf_limit < 0.01` raise `ValueError`
- **Opt-in SpeciesNet integration test** (marker `speciesnet_integration`, the same marker and env vars as `smartrodent_experiments`).
  It is skipped unless `RUN_SPECIESNET_INTEGRATION=1` is set, so it never runs in CI. `SPECIESNET_MODEL` may point to a local folder;
  otherwise the default `v4.0.3b` is loaded from the Kaggle cache, which is already populated locally. It checks real inference:
  - `tests/data/rodent.jpg` (copied from `smartrodent_experiments/resources/rodent.jpg`, 144 KB) gives at least one animal box
  - ultralytics' bundled `zidane.jpg` (people only) gives no boxes at `conf_limit: 0.5`. A moderate limit avoids flakiness from
    low-confidence false-positive animals near the 0.01 floor.
- Coverage: `pytest --cov=ml_pipeline --cov-report=term-missing --cov-branch`.

## 5. Out of scope
- Wiring the models into `services/` or the API. That includes deciding how class-agnostic boxes fill the `Detection`
  schema's `detected_class` field: the schema is left unchanged in this task.
- Kaggle downloads for anything other than SpeciesNet (where the SDK handles them). A state_dict + model-class loader (a possible later pickle-free option).
- Cropping detections for classification, and NMS beyond what each backend does.
- Any GPU support (R11: CPU only).

## 6. Decisions (all made by the user)
- D1 (was Q2): **decided by the user (R12).** Detectors: `(list[Tensor], list[list[BoundingBox]])`. Classifiers: `(list[Tensor], list[list[Classification]])`.
- D2: `speciesnet` is a regular dependency; the model comes from Kaggle (R14).
- D3 (was Q4):
  - no `device` key, CPU is hard-coded (R11); `batch_size` is optional (R15)
  - `apply_softmax` for `TorchClassifier` is kept (R16)
  - the `Taxon` base schema: accepted
  - `size` kwargs instead of `image_size`: accepted
  - removing `from_hub` / `_load_*_config`: accepted
- D4 (was Q5): **decided by the user.** The plan defines only the model's input/output contract (3.4) and nothing about how a model is produced.
- D5: SpeciesNet requires the `preprocessing` key with an empty value: accepted, and it must be documented (class docstring and config example).

## 7. Implementation order (after approval)
1. Tests stage: write `tests/test_model.py`, `tests/tiny_models.py`, the SpeciesNet stub, `tests/data/rodent.jpg` and the fixtures;
   register the `speciesnet_integration` marker in `pyproject.toml`; add the deps, and run `uv lock` and include `uv.lock`
   (CI runs `uv sync --locked`). The tests fail as expected. Review agent. Your approval.
2. Implementation stage: all tests pass. Review agent. Your approval.
3. Refactoring stage: tests stay green. Review agent. Your approval.
