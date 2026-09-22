from pydantic import BaseModel

from app.schemas.prediction import PredictionListItem


class ConfidenceBucket(BaseModel):
    range_label: str
    count: int


class BatchStats(BaseModel):
    total_batches: int
    pending: int
    processing: int
    completed: int
    failed: int
    total_images_via_batch: int


class AnalyticsSummary(BaseModel):
    total_predictions: int
    completed_predictions: int
    failed_predictions: int
    images_with_defects: int
    defect_rate: float
    defects_by_category: dict[str, int]
    confidence_distribution: list[ConfidenceBucket]
    recent_predictions: list[PredictionListItem]
    batch_stats: BatchStats
