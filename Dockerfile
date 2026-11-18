# Use official Python slim base image
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
# --no-deps: requirements.txt is the complete, pinned export of uv.lock; letting pip resolve
# dependencies itself would bring back opencv-python, which uv.lock deliberately overrides
RUN pip install --no-cache-dir --no-deps -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["python", "-m", "ml_pipeline.main"]
