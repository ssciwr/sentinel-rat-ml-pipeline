# Sentinel Rat ML Pipeline

Work in progress...

## Overview

FastAPI microservice for animal detection and species classification.

## API endpoints

- `POST /api/v1/analyze` — Analyze a batch of images
- `GET /health` — Health check

### Request

A non-empty list of image paths:

```json
{
  "image_paths": ["CAM01_image_20260728T170840Z.jpg", "CAM01_image_20260728T170905Z.jpg"]
}
```

### Response

One result per requested image, in request order. Each result holds one entry in `detections` per detected object, each with zero or more species `classifications`. Only detections of class `animal` are classified; others (e.g. `person`) have an empty list. Bounding boxes are relative to the image size (0-1).

```json
{
  "results": [
    {
      "image_path": "CAM01_image_20260728T170840Z.jpg",
      "detection_model": {"name": "megadetector", "version": "v5a", "description": "..."},
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
      ]
    },
    ...
  ]
}
```

## Test

```bash
pytest
```

## Example run

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/fake/a.jpg", "/fake/b.jpg"]}'
```

The service currently returns simulated results in the format above. Each image has one `animal` detection classified as black rat (0.87) or brown rat (0.10), with full taxonomy, and one unclassified `person` detection. Detection (`services/detection.py`) and classification (`services/classification.py`) are separate steps, combined per image in `services/analysis.py`.
