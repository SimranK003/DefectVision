def test_model_endpoint_503_when_nothing_registered(client):
    response = client.get("/model")
    assert response.status_code == 503


def test_model_endpoint_returns_production_model_info(client, production_model):
    response = client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "v1"
    assert body["dataset_version"] == "1.0.0"
    assert body["val_metrics"]["mAP50"] == 0.9
    assert "crazing" in body["classes"]


def test_model_endpoint_ignores_non_production_versions(client, db_session, production_model):
    from datetime import datetime, timezone

    from app.models.model_version import ModelVersion

    older = ModelVersion(
        version="v0",
        dataset_version="1.0.0",
        weights_path="ml/runs/old/weights/best.pt",
        run_dir="ml/runs/old",
        hardware="CPU: test",
        val_metrics={"mAP50": 0.1},
        config={},
        is_production=False,
        trained_at=datetime.now(timezone.utc),
    )
    db_session.add(older)
    db_session.commit()

    response = client.get("/model")

    assert response.status_code == 200
    assert response.json()["version"] == "v1"
