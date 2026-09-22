import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.batch_job import BatchJob, BatchStatus
from app.models.prediction import Prediction, PredictionStatus
from app.schemas.batch import BatchStatusResponse, BatchSubmitResponse
from app.services.storage import UploadValidationError, save_upload
from app.workers.tasks import process_prediction

router = APIRouter(tags=["batch"])


@router.post("/predict/batch", response_model=BatchSubmitResponse, status_code=202)
async def submit_batch(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> BatchSubmitResponse:
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {len(files)} files, max is {settings.max_batch_files}",
        )

    batch_job = BatchJob(status=BatchStatus.PENDING, total_images=len(files))
    db.add(batch_job)
    db.commit()
    db.refresh(batch_job)

    prediction_ids: list[uuid.UUID] = []
    failed_uploads = 0
    for file in files:
        try:
            saved = await save_upload(file)
        except UploadValidationError:
            # Bad file in a batch shouldn't kill the whole submission - count
            # it as an immediate failure so the batch's totals stay honest.
            failed_uploads += 1
            continue

        prediction = Prediction(
            status=PredictionStatus.PENDING,
            original_filename=saved.original_filename,
            image_path=str(saved.path.relative_to(settings.upload_dir)),
            batch_job_id=batch_job.id,
        )
        db.add(prediction)
        db.flush()
        prediction_ids.append(prediction.id)

    if failed_uploads:
        batch_job.failed_images = failed_uploads

    if not prediction_ids:
        # Every file in the batch failed validation before any task was queued -
        # nothing will ever call back to flip the status, so resolve it now.
        batch_job.status = BatchStatus.FAILED
        batch_job.completed_at = datetime.now(timezone.utc)
    else:
        batch_job.status = BatchStatus.PROCESSING
    db.commit()

    for pid in prediction_ids:
        process_prediction.delay(str(pid))

    db.refresh(batch_job)
    return BatchSubmitResponse(batch_id=batch_job.id, total_images=batch_job.total_images, status=batch_job.status)


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(batch_id: uuid.UUID, db: Session = Depends(get_db)) -> BatchStatusResponse:
    job = db.get(BatchJob, batch_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return BatchStatusResponse.from_orm_job(job)
