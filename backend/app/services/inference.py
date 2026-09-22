"""Loads the production YOLO model exactly once per process and reuses it
for every request. FastAPI holds one instance in app.state (see main.py's
lifespan); the Celery worker holds its own instance per worker process
(see workers/tasks.py) - either way, the weights are read from disk once,
not on every /predict call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from app.core.config import settings


class ModelNotAvailableError(RuntimeError):
    """Raised when no production model is registered yet."""


@dataclass
class Detection:
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class InferenceResult:
    detections: list[Detection]
    annotated_image_path: Path
    inference_time_ms: float
    model_version: str


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_production_entry() -> dict:
    if not settings.model_registry_path.exists():
        raise ModelNotAvailableError(
            f"No model registry at {settings.model_registry_path}. "
            "Run ml/scripts/train.py and ml/scripts/promote_model.py first."
        )
    entries = json.loads(settings.model_registry_path.read_text())
    for entry in entries:
        if entry.get("is_production"):
            return entry
    raise ModelNotAvailableError(
        "No model is marked is_production in the registry. "
        "Run ml/scripts/promote_model.py <version> to promote one."
    )


class InferenceEngine:
    """One YOLO model, loaded once, reused across requests."""

    def __init__(self) -> None:
        entry = _load_production_entry()
        repo_root = settings.model_registry_path.resolve().parents[2]
        weights_path = (
            settings.model_weights_override
            if settings.model_weights_override
            else repo_root / entry["weights_path"]
        )
        if not Path(weights_path).exists():
            raise ModelNotAvailableError(f"Weights file not found: {weights_path}")

        self.version = entry["version"]
        self.device = _resolve_device(settings.inference_device)
        self.model = YOLO(str(weights_path))
        self.model.to(self.device)
        self.class_names: dict[int, str] = self.model.names

    def predict(self, image_path: Path, output_dir: Path) -> InferenceResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        detections: list[Detection] = []
        img_h, img_w = result.orig_shape
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            detections.append(
                Detection(
                    class_name=self.class_names[class_id],
                    confidence=float(box.conf[0].item()),
                    x_min=xyxy[0] / img_w,
                    y_min=xyxy[1] / img_h,
                    x_max=xyxy[2] / img_w,
                    y_max=xyxy[3] / img_h,
                )
            )

        annotated_bgr = result.plot()  # numpy array, boxes + labels drawn
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_bgr)

        return InferenceResult(
            detections=detections,
            annotated_image_path=annotated_path,
            inference_time_ms=elapsed_ms,
            model_version=self.version,
        )


_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
