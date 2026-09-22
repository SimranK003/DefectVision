from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import analytics, batch, health, model, predict, predictions
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.sync_registry import sync_registry_to_db
from app.services.inference import ModelNotAvailableError, get_engine

# Created eagerly (not in lifespan) because StaticFiles below checks the
# directory exists at mount time, which happens before lifespan startup runs.
settings.upload_dir.mkdir(parents=True, exist_ok=True)
(settings.upload_dir / "originals").mkdir(exist_ok=True)
(settings.upload_dir / "annotated").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        sync_registry_to_db(db)
    finally:
        db.close()

    try:
        get_engine()  # load the model once, at process startup - not per-request
    except ModelNotAvailableError as exc:
        # Don't crash the whole API if no model has been trained/promoted yet -
        # /health and /model will report it, and /predict will fail loudly and
        # explain why, which is more useful than the API refusing to boot.
        print(f"[startup] WARNING: {exc}")

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.upload_dir), name="static")

if settings.ml_data_dir.exists():
    # Read-only: lets the frontend hotlink dataset/evaluation report images
    # (class distribution chart, confusion matrix, PR curves, sample
    # predictions) straight from where ml/scripts/*.py write them, instead
    # of duplicating those files into the upload directory.
    app.mount("/ml-static", StaticFiles(directory=settings.ml_data_dir), name="ml-static")

app.include_router(health.router)
app.include_router(model.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(predictions.router)
app.include_router(analytics.router)
