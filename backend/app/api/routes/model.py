from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.model_version import ModelVersion
from app.schemas.model_info import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def get_production_model(db: Session = Depends(get_db)) -> ModelInfoResponse:
    entry = db.query(ModelVersion).filter_by(is_production=True).one_or_none()
    if entry is None:
        raise HTTPException(status_code=503, detail="No production model is currently registered")

    classes = list(entry.config.get("classes") or []) if entry.config else []

    return ModelInfoResponse(
        version=entry.version,
        dataset_version=entry.dataset_version,
        trained_at=entry.trained_at,
        hardware=entry.hardware,
        val_metrics=entry.val_metrics,
        test_metrics=entry.test_metrics,
        config=entry.config,
        classes=classes,
    )
