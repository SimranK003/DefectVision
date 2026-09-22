"""Dashboard analytics computed live from the predictions/detections/batch_jobs
tables. Every number here is a real aggregate query - nothing is mocked, and
an empty database yields an empty-but-well-formed response (zeros, empty
lists), not fabricated sample data.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.batch_job import BatchJob, BatchStatus
from app.models.detection import Detection
from app.models.prediction import Prediction, PredictionStatus
from app.schemas.analytics import AnalyticsSummary, BatchStats, ConfidenceBucket
from app.schemas.prediction import PredictionListItem

router = APIRouter(tags=["analytics"])

CONFIDENCE_BUCKETS = [
    (0.0, 0.5, "0-50%"),
    (0.5, 0.7, "50-70%"),
    (0.7, 0.9, "70-90%"),
    (0.9, 1.01, "90-100%"),
]


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(db: Session = Depends(get_db)) -> AnalyticsSummary:
    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0
    completed_predictions = (
        db.query(func.count(Prediction.id)).filter(Prediction.status == PredictionStatus.COMPLETED).scalar() or 0
    )
    failed_predictions = (
        db.query(func.count(Prediction.id)).filter(Prediction.status == PredictionStatus.FAILED).scalar() or 0
    )

    images_with_defects = (
        db.query(func.count(func.distinct(Detection.prediction_id))).scalar() or 0
    )
    defect_rate = (images_with_defects / completed_predictions) if completed_predictions else 0.0

    defects_by_category = dict(
        db.query(Detection.class_name, func.count(Detection.id)).group_by(Detection.class_name).all()
    )

    confidence_buckets = []
    for low, high, label in CONFIDENCE_BUCKETS:
        count = (
            db.query(func.count(Detection.id))
            .filter(Detection.confidence >= low, Detection.confidence < high)
            .scalar()
            or 0
        )
        confidence_buckets.append(ConfidenceBucket(range_label=label, count=count))

    recent = (
        db.query(Prediction)
        .options(selectinload(Prediction.detections), selectinload(Prediction.model_version))
        .order_by(Prediction.created_at.desc())
        .limit(10)
        .all()
    )
    recent_items = []
    for p in recent:
        top = max(p.detections, key=lambda d: d.confidence, default=None)
        recent_items.append(
            PredictionListItem(
                id=p.id,
                status=p.status,
                original_filename=p.original_filename,
                inference_time_ms=p.inference_time_ms,
                model_version=p.model_version.version if p.model_version else None,
                created_at=p.created_at,
                num_detections=len(p.detections),
                top_class=top.class_name if top else None,
                top_confidence=top.confidence if top else None,
            )
        )

    batch_counts = dict(
        db.query(BatchJob.status, func.count(BatchJob.id)).group_by(BatchJob.status).all()
    )
    total_images_via_batch = db.query(func.coalesce(func.sum(BatchJob.total_images), 0)).scalar() or 0

    batch_stats = BatchStats(
        total_batches=sum(batch_counts.values()),
        pending=batch_counts.get(BatchStatus.PENDING, 0),
        processing=batch_counts.get(BatchStatus.PROCESSING, 0),
        completed=batch_counts.get(BatchStatus.COMPLETED, 0),
        failed=batch_counts.get(BatchStatus.FAILED, 0),
        total_images_via_batch=total_images_via_batch,
    )

    return AnalyticsSummary(
        total_predictions=total_predictions,
        completed_predictions=completed_predictions,
        failed_predictions=failed_predictions,
        images_with_defects=images_with_defects,
        defect_rate=defect_rate,
        defects_by_category=defects_by_category,
        confidence_distribution=confidence_buckets,
        recent_predictions=recent_items,
        batch_stats=batch_stats,
    )
