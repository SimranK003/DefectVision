import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.prediction import PredictionStatus
from app.schemas.detection import DetectionOut


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: PredictionStatus
    original_filename: str
    annotated_image_url: str | None = None
    inference_time_ms: float | None = None
    model_version: str | None = None
    error_message: str | None = None
    created_at: datetime
    detections: list[DetectionOut] = []


class PredictionListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: PredictionStatus
    original_filename: str
    inference_time_ms: float | None = None
    model_version: str | None = None
    created_at: datetime
    num_detections: int
    top_class: str | None = None
    top_confidence: float | None = None


class PredictionListResponse(BaseModel):
    items: list[PredictionListItem]
    total: int
    limit: int
    offset: int
