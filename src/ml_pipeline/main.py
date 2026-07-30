"""ML Pipeline - Animal detection and species classification service."""

from fastapi import FastAPI
from ml_pipeline.config import settings
from ml_pipeline.api.routes import router as api_router

app = FastAPI(title="ML Pipeline", version="0.1.0")
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ml-pipeline", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn  # type: ignore[import-untyped]

    uvicorn.run(app, host="0.0.0.0", port=settings.ML_PIPELINE_PORT)
