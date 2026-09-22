import asyncio
import io

import pytest
from app.services.storage import UploadValidationError, save_upload
from fastapi import UploadFile


def _upload_file(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content), headers={"content-type": content_type})


def test_rejects_disallowed_content_type(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    upload = _upload_file("doc.pdf", b"%PDF-1.4", "application/pdf")

    with pytest.raises(UploadValidationError, match="content type"):
        asyncio.run(save_upload(upload))


def test_rejects_disallowed_extension(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    upload = _upload_file("image.gif", b"GIF89a", "image/jpeg")  # spoofed content-type, bad extension

    with pytest.raises(UploadValidationError, match="extension"):
        asyncio.run(save_upload(upload))


def test_rejects_non_image_bytes_despite_correct_extension_and_type(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    upload = _upload_file("fake.jpg", b"this is definitely not jpeg data", "image/jpeg")

    with pytest.raises(UploadValidationError, match="not a valid image"):
        asyncio.run(save_upload(upload))

    # The invalid file must not be left behind on disk.
    assert list((tmp_path / "originals").glob("*")) == []


def test_rejects_oversized_file(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(config.settings, "max_upload_size_bytes", 100)
    upload = _upload_file("big.jpg", b"x" * 1000, "image/jpeg")

    with pytest.raises(UploadValidationError, match="max upload size"):
        asyncio.run(save_upload(upload))


def test_accepts_valid_jpeg(tmp_path, monkeypatch, sample_jpeg_bytes):
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    upload = _upload_file("part.jpg", sample_jpeg_bytes, "image/jpeg")

    saved = asyncio.run(save_upload(upload))

    assert saved.path.exists()
    assert saved.original_filename == "part.jpg"
    assert saved.size_bytes == len(sample_jpeg_bytes)


def test_uses_server_generated_filename_not_client_path(tmp_path, monkeypatch, sample_jpeg_bytes):
    """A malicious filename must never influence where the file lands on disk."""
    from app.core import config

    monkeypatch.setattr(config.settings, "upload_dir", tmp_path)
    upload = _upload_file("../../../etc/passwd.jpg", sample_jpeg_bytes, "image/jpeg")

    saved = asyncio.run(save_upload(upload))

    assert saved.path.parent == tmp_path / "originals"
    assert ".." not in str(saved.path)
