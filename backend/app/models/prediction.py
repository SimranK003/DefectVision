from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.batch_job import BatchJob
    from app.models.detection import Detection
    from app.models.model_version import ModelVersion


class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus, name="prediction_status"), default=PredictionStatus.PENDING, nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    annotated_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", index=True)

    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True
    )
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")

    batch_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=True, index=True
    )
    batch_job: Mapped["BatchJob"] = relationship(back_populates="predictions")

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )
