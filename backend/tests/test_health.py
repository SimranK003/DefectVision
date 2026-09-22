def test_health_reports_ok_database_and_redis_even_without_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["redis"] == "ok"
    # No model has been registered in this test DB, so overall status degrades,
    # but the endpoint itself must not error.
    assert body["status"] == "degraded"
    assert "error" in body["checks"]["model"]


def test_health_is_ok_once_a_production_model_and_engine_exist(client, production_model, fake_engine):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "redis": "ok", "model": "ok"}
