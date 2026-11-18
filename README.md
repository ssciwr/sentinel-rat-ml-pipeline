# Sentinel Rat ML Pipeline

Work in progress...

## Overview

FastAPI microservice for animal detection and species classification.

## API endpoints

- `POST /api/v1/analyze` — Analyze a batch of images
- `GET /api/v1/session` — Model cards of the running models
- `POST /api/v1/session` — Start a new session (load or reload the models)
- `GET /health` — Health check

### Request

A non-empty list of absolute image paths (a relative path is rejected with 422):

```json
{
  "image_paths": ["/data/CAM01/CAM01_image_20260728T170840Z.jpg", "/data/CAM01/CAM01_image_20260728T170905Z.jpg"]
}
```

### Response

One result per requested image, in request order. Each result holds one entry in `detections` per detected object, each with zero or more species `classifications`. `detected_class` is the detector's label (SpeciesNet: `animal`, `human` or `vehicle`), optionally renamed by the detector config's `label_map`. Only detections of class `animal` are classified; others (e.g. `human`) have an empty list. Bounding boxes are relative to the image size (0-1). Taxonomy fields the classifier does not know are `null`; this includes `genus` for family- or order-level classes such as SpeciesNet's `rodent`.

`error` is `null`, or a message if the image could not be loaded (missing file, not an image); such a result has empty `detections`, and the other images are analysed normally. Truncated images are analysed as far as they decode.

```json
{
  "results": [
    {
      "image_path": "/data/CAM01/CAM01_image_20260728T170840Z.jpg",
      "detection_model": {"name": "speciesnet-detector", "version": "v4.0.3b", "description": "..."},
      "classification_model": {"name": "rodent-classifier", "version": "0.1", "description": "..."},
      "detections": [
        {
          "detected_class": "animal",
          "confidence": 0.92,
          "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5},
          "classifications": [
            {
              "kingdom": null,
              "phylum": null,
              "class_name": null,
              "order": null,
              "family": null,
              "genus": "Rattus",
              "species": "rattus",
              "common_name": "black rat",
              "confidence": 0.87
            }
          ]
        }
      ],
      "error": null
    },
    ...
  ]
}
```

## Session and models

The running detector and classifier form the **session**. It is built on the first request that needs it, from the YAML config files named by the environment variables `DETECTOR_CONFIG` and `CLASSIFIER_CONFIG` (an empty value counts as unset), and reused afterwards. The service reads the process environment only, so load an env file yourself, e.g. `docker run --env-file .env ...` or `set -a; . ./.env; set +a` (`.env.example` points at the baseline configs below). If the session cannot be built (setting unset, invalid config, download failure), `GET /api/v1/session` and `/analyze` answer **503** with the reason, and the next request tries again.

`GET /api/v1/session` returns when the session was built and one model card per model:

```json
{
  "created_at": "2026-11-18T12:00:00Z",
  "models": [
    {"role": "detector", "type": "SpeciesnetDetector", "info": {"name": "speciesnet-detector", "version": "v4.0.3b", "description": "..."},
     "source": "configs/speciesnet_detector.yaml", "conf_limit": 0.1, "weights": "kaggle:google/speciesnet/pyTorch/v4.0.3b/1",
     "labels": ["animal", "human", "vehicle"]},
    {"role": "classifier", "type": "SpeciesnetClassifier", "info": {"name": "speciesnet-classifier", "version": "v4.0.3a", "description": "..."},
     "source": "configs/speciesnet_classifier.yaml", "conf_limit": 0.05, "weights": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1",
     "num_classes": 2498}
  ]
}
```

`source` is the config file, or `request` for an uploaded config; `weights` is the `model_path`, or `hf:<repo>/<file>[@<revision>]` for Hugging Face Hub weights. Detector cards list the labels the detector returns (after `label_map`), classifier cards the number of classes.

`POST /api/v1/session` builds a new session and replaces the running one. Its optional body holds complete model configs (the YAML keys below, as JSON objects) for `detector` and/or `classifier`; a role without a config (or a request without a body) is built from its environment variable, re-reading the config file. Configs may point at Hugging Face Hub weights (`hf_repo`, `hf_filename`, `hf_revision`), which the server downloads.

```json
{"classifier": {"type": "TorchClassifier", "info": {"name": "rodent-classifier", "version": "0.2"},
                "hf_repo": "org/rodent-models", "hf_filename": "classifier.pt", "conf_limit": 0.25,
                "preprocessing": {"resize": {"size": [448, 448]}}, "classes": [...], "apply_softmax": true}}
```

The new session is fully built before it replaces the old one: an invalid config (or an unknown role) gives **422**, `hf_repo` weights that cannot be downloaded from the Hugging Face Hub give **502**, and in both cases the old session keeps running. Download failures inside the SpeciesNet SDK (Kaggle, GitHub, `hf:` model ids) count as config errors (**422**). The `detail` of a 422 is a message for config errors and FastAPI's list of validation errors for a malformed body (e.g. an unknown role). The response is the new session's model cards.

**Security:** an uploaded config decides what the server downloads and unpickles, and torch models are loaded with `weights_only=False`, which can run code from the file. The session endpoint must only be reachable by the trusted client.

For every requested image, `services/analysis.py` loads the image once, runs the detector on it (`services/detection.py`), crops each `animal` detection and classifies the crop (`services/classification.py`). Images are processed one at a time, on CPU.

Config keys:

```yaml
type: TorchClassifier              # YoloDetector | SpeciesnetDetector | TorchDetector | YoloClassifier | SpeciesnetClassifier | TorchClassifier
info:                              # reported as detection_model / classification_model
  name: rodent-classifier
  version: "0.1"
  description: Rodent species classifier
model_path: /etc/ml/rodent-cls.pt  # relative paths resolve against the server's working directory;
                                   # or hf_repo + hf_filename (+ hf_revision) for the Hugging Face Hub
conf_limit: 0.25                   # [0, 1]; SpeciesNet: [0.01, 1]
preprocessing: {}                  # required; torchvision v2 transforms, e.g. {resize: {size: [448, 448]}}; SpeciesNet: empty
classes:                           # classifiers: one taxonomy entry per model output, in order
  - {genus: Rattus, species: norvegicus, common_name: brown rat}
apply_softmax: true                # TorchClassifier: true if the model outputs logits
# labels: [animal, human, vehicle] # TorchDetector: label of each class index
# label_map: {cat: animal}         # detectors, optional: rename model labels
```

- `SpeciesnetDetector` uses SpeciesNet's detector (MegaDetector v5a); `model_path` defaults to `kaggle:google/speciesnet/pyTorch/v4.0.3b/1`, which the SDK downloads from Kaggle on first use.
- `SpeciesnetClassifier` uses SpeciesNet's crop classifier on each crop; `model_path` defaults to `kaggle:google/speciesnet/pyTorch/v4.0.3a/1` (only crop classifiers are accepted). It has no `classes` key: the taxonomy comes from the model's label file. Its output is the raw classifier, without SpeciesNet's geofencing and taxonomic rollup; a few classes share a taxonomy (they differ only in SpeciesNet's internal id), so one crop can get two classifications with the same taxonomy.
- SpeciesNet `model_path`s may be a local model folder, a `kaggle:` id or an `hf:` id (all handled by the SDK; `hf_repo`/`hf_filename` are not supported because a SpeciesNet model is a folder). Both SpeciesNet models also fetch the MegaDetector v5a weights from GitHub on first use, as listed in their `info.json`.
- YOLO models are ultralytics weights; `model_path` may be an ultralytics asset name such as `yolo26n.pt`, which ultralytics downloads.
- Torch models are whole pickled `nn.Module`s (`torch.save(model)`), loaded with `weights_only=False`: load trusted files only. A classifier must return `[1, C]` logits or probabilities, a detector `[1, K, 6]` rows `(x1, y1, x2, y2, score, class_index)` with relative corners; rows with score 0 are padding.

### Baseline configs

`configs/speciesnet_detector.yaml` (SpeciesNet v4.0.3b detector) and `configs/speciesnet_classifier.yaml` (SpeciesNet v4.0.3a crop classifier) form a validation baseline. The SDK downloads both models from Kaggle (and the MegaDetector weights from GitHub) on first use.

## Test

```bash
pytest
```

The YOLO tests download two small pretrained models into the pytest cache on first use. The real SpeciesNet tests are opt-in: `RUN_SPECIESNET_INTEGRATION=1 pytest`.

## Example run

```bash
DETECTOR_CONFIG=configs/speciesnet_detector.yaml CLASSIFIER_CONFIG=configs/speciesnet_classifier.yaml python -m ml_pipeline.main

curl http://127.0.0.1:8000/api/v1/session

curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/data/a.jpg", "/data/b.jpg"]}'
```
