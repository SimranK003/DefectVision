from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    version: str
    dataset_version: str
    trained_at: datetime
    hardware: str
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    config: dict[str, Any]
    classes: list[str]
