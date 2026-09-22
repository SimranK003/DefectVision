import uuid

from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
