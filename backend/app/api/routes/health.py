from fastapi import APIRouter, Depends
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    checks = {"database": "ok", "redis": "ok", "model": "ok"}

    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - health check must not leak internals
        checks["database"] = f"error: {type(exc).__name__}"

    try:
        Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
    except RedisError as exc:
        checks["redis"] = f"error: {type(exc).__name__}"

    try:
        from app.services.inference import get_engine

        get_engine()
    except Exception as exc:  # noqa: BLE001
        checks["model"] = f"error: {exc}"

    overall_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if overall_ok else "degraded", "checks": checks}
