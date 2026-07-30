# Sentinel Rat ML Pipeline

Work in progress...

## Overview

FastAPI microservice for animal detection and species classification.

## API endpoints

- `POST /api/v1/analyze` — Analyze an image
- `GET /health` — Health check

### Request

```json
{
  "image_path": "/path/to/image.jpg"
}
```

### Response

```json
{
  "image_path": "/path/to/image.jpg",
  "animals_detected": 2,
  "species": {"cat": 1, "rodent": 1},
  "confidence": 0.95
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
  -d '{"image_path": "/fake/path.jpg"}'
```

Expected response:

```json
{
  "image_path": "/fake/path.jpg",
  "animals_detected": 2,
  "species": {"cat": 1, "rodent": 1},
  "confidence": 0.95
}
```
