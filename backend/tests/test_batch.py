import pytest
from app.workers.celery_app import celery_app


@pytest.fixture
def eager_celery():
    """Run Celery tasks synchronously, in-process - no Redis broker or
    separate worker needed for the test to exercise the real task code."""
    original_eager = celery_app.conf.task_always_eager
    original_propagate = celery_app.conf.task_eager_propagates
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = original_eager
    celery_app.conf.task_eager_propagates = original_propagate


def test_submit_batch_creates_job_and_processes_synchronously(
    client, production_model, fake_engine, eager_celery, sample_jpeg_bytes
):
    files = [("files", (f"part_{i}.jpg", sample_jpeg_bytes, "image/jpeg")) for i in range(3)]

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 202
    body = response.json()
    assert body["total_images"] == 3
    batch_id = body["batch_id"]

    status_response = client.get(f"/batch/{batch_id}")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "completed"
    assert status_body["completed_images"] == 3
    assert status_body["failed_images"] == 0
    assert status_body["completed_at"] is not None


def test_batch_partial_failure_still_completes(
    client, production_model, fake_engine, eager_celery, sample_jpeg_bytes
):
    files = [
        ("files", ("good.jpg", sample_jpeg_bytes, "image/jpeg")),
        ("files", ("bad.txt", b"not an image", "image/jpeg")),  # wrong bytes, right content-type
    ]

    response = client.post("/predict/batch", files=files)
    batch_id = response.json()["batch_id"]

    status = client.get(f"/batch/{batch_id}").json()
    assert status["total_images"] == 2
    assert status["completed_images"] == 1
    assert status["failed_images"] == 1
    assert status["status"] == "completed"  # partial success still resolves as completed


def test_batch_all_files_invalid_resolves_as_failed(client, production_model, fake_engine, eager_celery):
    files = [("files", ("bad.txt", b"not an image", "image/jpeg"))]

    response = client.post("/predict/batch", files=files)
    batch_id = response.json()["batch_id"]

    status = client.get(f"/batch/{batch_id}").json()
    assert status["status"] == "failed"
    assert status["failed_images"] == 1
    assert status["completed_at"] is not None


def test_batch_rejects_empty_submission(client):
    response = client.post("/predict/batch", files=[])
    assert response.status_code == 422


def test_batch_rejects_over_limit_submission(client, monkeypatch, sample_jpeg_bytes):
    from app.core import config

    monkeypatch.setattr(config.settings, "max_batch_files", 2)
    files = [("files", (f"part_{i}.jpg", sample_jpeg_bytes, "image/jpeg")) for i in range(3)]

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 422


def test_unknown_batch_id_returns_404(client):
    response = client.get("/batch/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
