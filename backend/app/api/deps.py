"""FastAPI dependency wrappers.

Routes depend on `get_inference_engine` (not `app.services.inference.get_engine`
directly) so tests can override it via `app.dependency_overrides` instead of
needing a real trained model on disk.
"""

from app.services.inference import InferenceEngine, get_engine


def get_inference_engine() -> InferenceEngine:
    return get_engine()
