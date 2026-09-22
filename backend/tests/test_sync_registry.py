import json
from pathlib import Path

from app.db.sync_registry import sync_registry_to_db
from app.models.model_version import ModelVersion


def _write_registry(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def test_sync_creates_rows_from_registry_file(tmp_path, db_session, monkeypatch):
    from app.core import config

    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            {
                "version": "v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "dataset_version": "1.0.0",
                "weights_path": "ml/runs/v1/weights/best.pt",
                "run_dir": "ml/runs/v1",
                "hardware": "CPU",
                "val_metrics": {"mAP50": 0.5},
                "test_metrics": None,
                "config": {"classes": ["crazing"]},
                "is_production": True,
            }
        ],
    )
    monkeypatch.setattr(config.settings, "model_registry_path", registry_path)

    sync_registry_to_db(db_session)

    row = db_session.query(ModelVersion).filter_by(version="v1").one()
    assert row.dataset_version == "1.0.0"
    assert row.is_production is True
    assert row.val_metrics == {"mAP50": 0.5}


def test_sync_is_idempotent_and_updates_existing_rows(tmp_path, db_session, monkeypatch):
    from app.core import config

    registry_path = tmp_path / "registry.json"
    entries = [
        {
            "version": "v1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "dataset_version": "1.0.0",
            "weights_path": "ml/runs/v1/weights/best.pt",
            "run_dir": "ml/runs/v1",
            "hardware": "CPU",
            "val_metrics": {"mAP50": 0.5},
            "test_metrics": None,
            "config": {},
            "is_production": False,
        }
    ]
    _write_registry(registry_path, entries)
    monkeypatch.setattr(config.settings, "model_registry_path", registry_path)

    sync_registry_to_db(db_session)
    assert db_session.query(ModelVersion).count() == 1

    # Re-sync after the entry gets test_metrics and gets promoted - same
    # version should update in place, not create a duplicate row.
    entries[0]["test_metrics"] = {"mAP50": 0.55}
    entries[0]["is_production"] = True
    _write_registry(registry_path, entries)

    sync_registry_to_db(db_session)

    assert db_session.query(ModelVersion).count() == 1
    row = db_session.query(ModelVersion).filter_by(version="v1").one()
    assert row.test_metrics == {"mAP50": 0.55}
    assert row.is_production is True


def test_sync_handles_multiple_versions_with_one_production(tmp_path, db_session, monkeypatch):
    from app.core import config

    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            {
                "version": "v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "dataset_version": "1.0.0",
                "weights_path": "a",
                "run_dir": "a",
                "hardware": "CPU",
                "val_metrics": {},
                "test_metrics": None,
                "config": {},
                "is_production": False,
            },
            {
                "version": "v2",
                "created_at": "2026-01-02T00:00:00+00:00",
                "dataset_version": "1.0.0",
                "weights_path": "b",
                "run_dir": "b",
                "hardware": "CPU",
                "val_metrics": {},
                "test_metrics": None,
                "config": {},
                "is_production": True,
            },
        ],
    )
    monkeypatch.setattr(config.settings, "model_registry_path", registry_path)

    sync_registry_to_db(db_session)

    prod = db_session.query(ModelVersion).filter_by(is_production=True).all()
    assert [p.version for p in prod] == ["v2"]


def test_sync_no_op_when_registry_file_missing(db_session, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "model_registry_path", Path("/tmp/does-not-exist-registry.json"))

    sync_registry_to_db(db_session)  # must not raise

    assert db_session.query(ModelVersion).count() == 0
