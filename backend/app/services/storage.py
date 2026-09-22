"""Safe handling of user-uploaded images.

Defense in depth against a malicious/malformed upload:
  1. reject on declared Content-Type up front (cheap, not sufficient alone)
  2. enforce a hard size cap while streaming to disk (no unbounded reads)
  3. re-verify the extension against an allowlist (never trust the client's
     filename for anything beyond display)
  4. decode the saved bytes with Pillow and call .verify() - this is what
     actually catches a renamed non-image file, a truncated file, or a
     polyglot; a file that fails this is deleted, not processed further
  5. write to disk under a server-generated UUID filename - the original
     filename is stored only as metadata, never used to build a path, so a
     "../../etc/passwd.jpg" filename can't escape upload_dir.

We never execute, import, or eval anything from the upload - it is only
ever opened as pixel data (PIL / OpenCV / Ultralytics), which is what makes
"never execute uploaded files" true structurally, not just by policy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import settings


class UploadValidationError(ValueError):
    """Raised for any upload that fails validation. Message is safe to show the client."""


@dataclass
class SavedUpload:
    path: Path
    original_filename: str
    size_bytes: int


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UploadValidationError(
            f"Unsupported file extension '{suffix}'. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    return suffix


async def save_upload(file: UploadFile, subdir: str = "originals") -> SavedUpload:
    if file.content_type not in settings.allowed_content_types:
        raise UploadValidationError(
            f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(settings.allowed_content_types)}"
        )

    suffix = _safe_suffix(file.filename or "")

    dest_dir = settings.upload_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{uuid.uuid4().hex}{suffix}"

    size = 0
    chunk_size = 1024 * 1024
    with open(dest_path, "wb") as out:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > settings.max_upload_size_bytes:
                out.close()
                dest_path.unlink(missing_ok=True)
                raise UploadValidationError(
                    f"File exceeds max upload size of {settings.max_upload_size_bytes // (1024 * 1024)} MB"
                )
            out.write(chunk)

    if size == 0:
        dest_path.unlink(missing_ok=True)
        raise UploadValidationError("Uploaded file is empty")

    try:
        with Image.open(dest_path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        dest_path.unlink(missing_ok=True)
        raise UploadValidationError("File is not a valid image") from exc

    return SavedUpload(path=dest_path, original_filename=file.filename or dest_path.name, size_bytes=size)
