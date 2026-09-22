"""Environment-based configuration. Never hardcode secrets or paths here -
every field is overridable via env var / .env, so dev, CI, and Docker all
run the same code against different backing services."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "DefectVision API"
    environment: str = "development"  # development | test | production
    api_prefix: str = "/api/v1"

    # --- Database ---
    database_url: str = "postgresql+psycopg://defectvision:defectvision@localhost:5432/defectvision"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # --- Model ---
    model_registry_path: Path = REPO_ROOT / "ml/models/registry.json"
    ml_data_dir: Path = REPO_ROOT / "ml/data"  # served read-only at /ml-static (dataset + evaluation reports)
    model_weights_override: Path | None = None  # force a specific weights file, bypassing the registry
    # Measured, not assumed: for this small model (YOLOv8n, 3M params, 320px),
    # CPU serving latency (median ~20ms) beat MPS (median ~55ms) on this Apple
    # M4 - GPU dispatch overhead dominates for a model this lightweight. See
    # README "Performance". Override to "mps"/"cuda" if benchmarking shows
    # otherwise on different hardware or a larger model.
    inference_device: str = "cpu"  # auto | cpu | mps | cuda
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # --- Uploads / security ---
    upload_dir: Path = REPO_ROOT / "backend/uploads"
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB per image
    max_batch_files: int = 200
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png")
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # --- CORS ---
    cors_allow_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:3100")


settings = Settings()
