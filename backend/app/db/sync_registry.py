"""Upserts ml/models/registry.json entries into the model_versions table.

Run automatically on API startup (see main.py) so GET /model and the
predictions.model_version_id FK always reflect whatever's on disk, without
a separate manual DB-seeding step after every training run.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.model_version import ModelVersion


def sync_registry_to_db(db: Session) -> None:
    if not settings.model_registry_path.exists():
        return

    entries = json.loads(settings.model_registry_path.read_text())
    for entry in entries:
        existing = db.query(ModelVersion).filter_by(version=entry["version"]).one_or_none()
        if existing is None:
            existing = ModelVersion(version=entry["version"])
            db.add(existing)

        existing.dataset_version = entry["dataset_version"]
        existing.weights_path = entry["weights_path"]
        existing.run_dir = entry["run_dir"]
        existing.hardware = entry["hardware"]
        existing.val_metrics = entry["val_metrics"]
        existing.test_metrics = entry.get("test_metrics")
        existing.config = entry.get("config", {})
        existing.is_production = bool(entry.get("is_production", False))
        existing.trained_at = datetime.fromisoformat(entry["created_at"])

    db.commit()
