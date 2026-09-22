import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.prediction import Prediction, PredictionStatus
from app.schemas.detection import DetectionOut
from app.schemas.prediction import PredictionListItem, PredictionListResponse, PredictionOut

router = APIRouter(tags=["predictions"])


def _to_out(prediction: Prediction) -> PredictionOut:
    # See app.api.routes.predict._to_out for why this isn't model_validate().
    return PredictionOut(
        id=prediction.id,
        status=prediction.status,
        original_filename=prediction.original_filename,
        annotated_image_url=f"/static/{prediction.annotated_image_path}" if prediction.annotated_image_path else None,
        inference_time_ms=prediction.inference_time_ms,
        model_version=prediction.model_version.version if prediction.model_version else None,
        error_message=prediction.error_message,
        created_at=prediction.created_at,
        detections=[DetectionOut.model_validate(d) for d in prediction.detections],
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: uuid.UUID, db: Session = Depends(get_db)) -> PredictionOut:
    prediction = (
        db.query(Prediction)
        .options(selectinload(Prediction.detections), selectinload(Prediction.model_version))
        .filter(Prediction.id == prediction_id)
        .one_or_none()
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _to_out(prediction)


@router.get("/predictions", response_model=PredictionListResponse)
def list_predictions(
    status: PredictionStatus | None = None,
    batch_id: uuid.UUID | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    query = db.query(Prediction).options(
        selectinload(Prediction.detections), selectinload(Prediction.model_version)
    )
    if status is not None:
        query = query.filter(Prediction.status == status)
    if batch_id is not None:
        query = query.filter(Prediction.batch_job_id == batch_id)

    total = query.with_entities(func.count(Prediction.id)).scalar() or 0
    rows = query.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for p in rows:
        top_detection = max(p.detections, key=lambda d: d.confidence, default=None)
        items.append(
            PredictionListItem(
                id=p.id,
                status=p.status,
                original_filename=p.original_filename,
                inference_time_ms=p.inference_time_ms,
                model_version=p.model_version.version if p.model_version else None,
                created_at=p.created_at,
                num_detections=len(p.detections),
                top_class=top_detection.class_name if top_detection else None,
                top_confidence=top_detection.confidence if top_detection else None,
            )
        )

    return PredictionListResponse(items=items, total=total, limit=limit, offset=offset)
