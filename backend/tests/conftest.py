"""Shared test fixtures.

Tests run against a real Postgres database (defectvision_test) rather than
SQLite - the ORM models use Postgres-specific UUID/JSON/Enum types, so a
SQLite-backed test would be testing a different dialect than production.
Each test gets a clean slate via TRUNCATE, not a fresh schema, which keeps
the suite fast.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://defectvision:defectvision@localhost:5432/defectvision_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
# Point at a registry file that doesn't exist so app startup never syncs the
# *real* trained-model registry into the test DB - tests seed their own
# ModelVersion rows via the `production_model` fixture instead, keeping the
# suite independent of whatever has (or hasn't) actually been trained.
os.environ["MODEL_REGISTRY_PATH"] = "/tmp/defectvision_test_registry_does_not_exist.json"
os.environ["UPLOAD_DIR"] = "/tmp/defectvision_test_uploads"

from app.db.session import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.model_version import ModelVersion  # noqa: E402
from app.services.inference import Detection as InferDetection  # noqa: E402
from app.services.inference import InferenceResult  # noqa: E402

test_engine = create_engine(TEST_DATABASE_URL, future=True)
TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Truncate every table before each test so tests don't leak state."""
    with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" RESTART IDENTITY CASCADE'))
    yield


@pytest.fixture
def db_session():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def _get_test_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_test_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def production_model(db_session) -> ModelVersion:
    """A registered production model row - most endpoints assume one exists."""
    model = ModelVersion(
        version="v1",
        dataset_version="1.0.0",
        weights_path="ml/runs/defectvision/weights/best.pt",
        run_dir="ml/runs/defectvision",
        hardware="CPU: test",
        val_metrics={"mAP50": 0.9, "mAP50_95": 0.6, "precision": 0.8, "recall": 0.85},
        test_metrics={"mAP50": 0.88, "mAP50_95": 0.58, "precision": 0.79, "recall": 0.83},
        config={"classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]},
        is_production=True,
        trained_at=datetime.now(timezone.utc),
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


class FakeInferenceEngine:
    """Stands in for the real YOLO engine in tests: deterministic, instant,
    no dependency on a trained weights file being present."""

    version = "v1"

    def __init__(self, detections: list[InferDetection] | None = None):
        self._detections = detections if detections is not None else [
            InferDetection(class_name="crazing", confidence=0.91, x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        ]

    def predict(self, image_path: Path, output_dir: Path) -> InferenceResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / f"{Path(image_path).stem}_annotated.jpg"
        annotated_path.write_bytes(b"fake-annotated-image-bytes")
        return InferenceResult(
            detections=self._detections,
            annotated_image_path=annotated_path,
            inference_time_ms=12.3,
            model_version=self.version,
        )


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeInferenceEngine()
    monkeypatch.setattr("app.api.routes.predict.get_inference_engine", lambda: engine)
    monkeypatch.setattr("app.workers.tasks.get_engine", lambda: engine)
    # health.py does `from app.services.inference import get_engine` *inside*
    # the request handler, so patching the source attribute (rather than a
    # module that already bound its own copy of the name) is what's needed
    # for that call site to see the fake.
    monkeypatch.setattr("app.services.inference.get_engine", lambda: engine)
    return engine


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (200, 200), color=(120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()
