# Sentinel Rat ML Pipeline

Work in progress...

## Overview

FastAPI microservice for animal detection and species classification.

## API endpoints

- `POST /api/v1/analyze` — Analyze a batch of images
- `GET /health` — Health check

### Request

A non-empty list of absolute image paths (a relative path is rejected with 422):

```json
{
  "image_paths": ["/data/CAM01/CAM01_image_20260728T170840Z.jpg", "/data/CAM01/CAM01_image_20260728T170905Z.jpg"]
}
```

### Response

One result per requested image, in request order. Each result holds one entry in `detections` per detected object, each with zero or more species `classifications`. `detected_class` is the detector's label (SpeciesNet: `animal`, `human` or `vehicle`), optionally renamed by the detector config's `label_map`. Only detections of class `animal` are classified; others (e.g. `human`) have an empty list. Bounding boxes are relative to the image size (0-1).

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

## Models

The service uses one detector and one classifier, each described by a YAML config file. Set their paths in the environment variables `DETECTOR_CONFIG` and `CLASSIFIER_CONFIG`; an empty value counts as unset. The service reads the process environment only, so load an env file yourself, e.g. `docker run --env-file .env ...` or `set -a; . ./.env; set +a` (see `.env.example`). Each model is built on the first request that needs it and reused afterwards. Without a config, that request fails with HTTP 500, and the server log names the missing variable.

For every requested image, `services/analysis.py` loads the image once, runs the detector on it (`services/detection.py`), crops each `animal` detection and classifies the crop (`services/classification.py`). Images are processed one at a time, on CPU.

Config keys:

```yaml
type: TorchClassifier              # YoloDetector | SpeciesnetDetector | TorchDetector | YoloClassifier | TorchClassifier
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
- YOLO models are ultralytics weights; `model_path` may be an ultralytics asset name such as `yolo26n.pt`, which ultralytics downloads.
- Torch models are whole pickled `nn.Module`s (`torch.save(model)`), loaded with `weights_only=False`: load trusted files only. A classifier must return `[1, C]` logits or probabilities, a detector `[1, K, 6]` rows `(x1, y1, x2, y2, score, class_index)` with relative corners; rows with score 0 are padding.

## Test

```bash
pytest
```

The YOLO tests download two small pretrained models into the pytest cache on first use. The real SpeciesNet test is opt-in: `RUN_SPECIESNET_INTEGRATION=1 pytest`.

## Example run

```bash
DETECTOR_CONFIG=/etc/ml/detector.yaml CLASSIFIER_CONFIG=/etc/ml/classifier.yaml python -m ml_pipeline.main

curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/data/a.jpg", "/data/b.jpg"]}'
```
