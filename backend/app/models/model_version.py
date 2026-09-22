"""Mirrors an entry from ml/models/registry.json into the DB so the API can
serve /model without reading the filesystem on every request, and so
predictions can carry a foreign key to exactly which trained model produced
them."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.prediction import Prediction


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(32), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(512), nullable=False)
    run_dir: Mapped[str] = mapped_column(String(512), nullable=False)
    hardware: Mapped[str] = mapped_column(String(256), nullable=False)
    val_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    test_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")
