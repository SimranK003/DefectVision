from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_inference_engine
from app.core.config import settings
from app.db.session import get_db
from app.models.detection import Detection
from app.models.model_version import ModelVersion
from app.models.prediction import Prediction, PredictionStatus
from app.schemas.detection import DetectionOut
from app.schemas.prediction import PredictionOut
from app.services.inference import ModelNotAvailableError
from app.services.storage import UploadValidationError, save_upload

router = APIRouter(tags=["predict"])


def _to_out(prediction: Prediction) -> PredictionOut:
    # Built field-by-field rather than via PredictionOut.model_validate(prediction):
    # the ORM's `model_version` is a relationship (a ModelVersion object), but
    # the schema's `model_version` field is the version *string* - same name,
    # different shape, so automatic from_attributes mapping would try to
    # validate the relationship object as a string and fail.
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


@router.post("/predict", response_model=PredictionOut, status_code=201)
async def predict_single(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PredictionOut:
    try:
        saved = await save_upload(file)
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = Prediction(
        status=PredictionStatus.PROCESSING,
        original_filename=saved.original_filename,
        image_path=str(saved.path.relative_to(settings.upload_dir)),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    try:
        engine = get_inference_engine()
    except ModelNotAvailableError as exc:
        prediction.status = PredictionStatus.FAILED
        prediction.error_message = str(exc)
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = engine.predict(saved.path, output_dir=settings.upload_dir / "annotated")
    except Exception as exc:  # noqa: BLE001 - convert any inference failure into a stored, visible failure
        prediction.status = PredictionStatus.FAILED
        prediction.error_message = f"Inference failed: {exc}"
        db.commit()
        raise HTTPException(status_code=500, detail="Inference failed") from exc

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
    db.refresh(prediction)

    return _to_out(prediction)
