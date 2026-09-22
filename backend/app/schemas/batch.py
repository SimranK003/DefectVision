import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.batch_job import BatchStatus


class BatchSubmitResponse(BaseModel):
    batch_id: uuid.UUID
    total_images: int
    status: BatchStatus


class BatchStatusResponse(BaseModel):
    batch_id: uuid.UUID
    status: BatchStatus
    total_images: int
    completed_images: int
    failed_images: int
    created_at: datetime
    completed_at: datetime | None = None

    @classmethod
    def from_orm_job(cls, job) -> "BatchStatusResponse":
        return cls(
            batch_id=job.id,
            status=job.status,
            total_images=job.total_images,
            completed_images=job.completed_images,
            failed_images=job.failed_images,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )
