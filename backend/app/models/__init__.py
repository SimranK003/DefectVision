"""Import every ORM model here so Alembic's autogenerate and Base.metadata
see the full schema regardless of which module happens to be imported first."""

from app.models.batch_job import BatchJob, BatchStatus  # noqa: F401
from app.models.detection import Detection  # noqa: F401
from app.models.model_version import ModelVersion  # noqa: F401
from app.models.prediction import Prediction, PredictionStatus  # noqa: F401
