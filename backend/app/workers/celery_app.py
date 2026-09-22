from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "defectvision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one image at a time per worker process - inference is the bottleneck, not I/O
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)

celery_app.autodiscover_tasks(["app.workers"])
