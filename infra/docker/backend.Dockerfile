# Shared image for both the "api" and "worker" services (docker-compose.yml
# picks the command). Both need the exact same dependencies - FastAPI,
# SQLAlchemy, and the full torch/ultralytics stack to load the YOLO model -
# so one image avoids drifting requirement sets between them.
FROM python:3.11-slim AS base

# libgl1/libglib2.0-0: OpenCV's import chain still touches these even with
# opencv-python-headless, on Debian slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/base.txt requirements/backend.txt requirements/
RUN pip install --no-cache-dir -r requirements/backend.txt

COPY backend/ backend/
COPY ml/src ml/src
COPY ml/configs ml/configs

WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# docker-compose overrides `command:` for the worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
