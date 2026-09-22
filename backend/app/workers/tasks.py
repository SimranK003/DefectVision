"""Celery tasks: one task per image so batch progress is trackable and a
single bad image can't block the rest of the batch. The YOLO model is
loaded once per worker process via get_engine()'s module-level cache - the
first task on a given worker pays the load cost, every task after it doesn't.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from celery.utils.log import get_task_logger
from sqlalchemy import update

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.batch_job import BatchJob, BatchStatus
from app.models.detection import Detection
from app.models.model_version import ModelVersion
from app.models.prediction import Prediction, PredictionStatus
from app.services.inference import get_engine
from app.workers.celery_app import celery_app

logger = get_task_logger(__name__)


def _bump_batch_counts(db, batch_job_id: uuid.UUID | None, failed: bool) -> None:
    if batch_job_id is None:
        return

    column = BatchJob.failed_images if failed else BatchJob.completed_images
    db.execute(update(BatchJob).where(BatchJob.id == batch_job_id).values({column: column + 1}))
    db.commit()

    job = db.get(BatchJob, batch_job_id)
    if job is None:
        return
    attempted = job.completed_images + job.failed_images
    if attempted >= job.total_images:
        job.status = BatchStatus.FAILED if job.failed_images == job.total_images else BatchStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db.commit()


@celery_app.task(name="process_prediction")
def process_prediction(prediction_id: str) -> None:
    db = SessionLocal()
    try:
        prediction = db.get(Prediction, uuid.UUID(prediction_id))
        if prediction is None:
            logger.warning("process_prediction: prediction %s not found", prediction_id)
            return

        prediction.status = PredictionStatus.PROCESSING
        db.commit()

        image_path = settings.upload_dir / prediction.image_path

        try:
            engine = get_engine()
            result = engine.predict(image_path, output_dir=settings.upload_dir / "annotated")
        except Exception as exc:  # noqa: BLE001 - a per-image failure must not crash the batch
            logger.exception("inference failed for prediction %s", prediction_id)
            prediction.status = PredictionStatus.FAILED
            prediction.error_message = str(exc)[:1000]
            db.commit()
            _bump_batch_counts(db, prediction.batch_job_id, failed=True)
            return

        model_version_row = db.query(ModelVersion).filter_by(version=result.model_version).one_or_none()
        prediction.status = PredictionStatus.COMPLETED
        prediction.annotated_image_path = str(result.annotated_image_path.relative_to(settings.upload_dir))
        prediction.inference_time_ms = result.inference_time_ms
        prediction.model_version_id = model_version_row.id if model_version_row else None
        for det in result.detections:
            db.add(
                Detection(
                    prediction_id=prediction.id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    x_min=det.x_min,
                    y_min=det.y_min,
                    x_max=det.x_max,
                    y_max=det.y_max,
                    model_version=result.model_version,
                )
            )
        db.commit()
        _bump_batch_counts(db, prediction.batch_job_id, failed=False)
    finally:
        db.close()
