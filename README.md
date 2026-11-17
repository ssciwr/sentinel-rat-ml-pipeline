# Sentinel Rat ML Pipeline

Work in progress...

## Overview

FastAPI microservice for animal detection and species classification.

## API endpoints

- `POST /api/v1/analyze` — Analyze a batch of images
- `GET /health` — Health check

### Request

A non-empty list of image paths, relative to the watch folder. The service reads each image from `WATCH_FOLDER_DOCKER` (default `/data/images`), where the watch folder of sentinel-rat-pipeline is mounted read-only (see `docker-compose.yml` of sentinel-rat):

```json
{
  "image_paths": ["hd-zoo-01/image_20260728T170840Z.jpg", "hd-zoo-01/image_20260728T170905Z.jpg"]
}
```

### Response

One result per requested image, in request order. An image that can't be read (missing, or outside the watch folder) is logged and left out of the results, so the rest of the batch is still analyzed. Each result holds one entry in `detections` per detected object, each with zero or more species `classifications`. Only detections of class `animal` are classified; others (e.g. `person`) have an empty list. Bounding boxes are relative to the image size (0-1).

```json
{
  "results": [
    {
      "image_path": "hd-zoo-01/image_20260728T170840Z.jpg",
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

With an image at `$WATCH_FOLDER_DOCKER/hd-zoo-01/a.jpg`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["hd-zoo-01/a.jpg"]}'
```

The service reads each image but currently returns simulated results in the format above. Each image has one `animal` detection classified as black rat (0.87) or brown rat (0.10), with full taxonomy, and one unclassified `person` detection. Detection (`services/detection.py`) and classification (`services/classification.py`) are separate steps, combined per image in `services/analysis.py`.
