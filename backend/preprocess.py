"""Safe image validation and model input preprocessing."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = (299, 299)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class InvalidImageError(ValueError):
    """Raised when an upload cannot be processed as a supported image."""


async def validate_upload(upload: UploadFile) -> bytes:
    """Check MIME type, size, and image decodability before inference."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError("Upload a JPEG, PNG, or WebP image.")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise InvalidImageError("The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error
    return data


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert an upload to RGB 299×299, normalize to [0, 1], and batch it."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error

    normalized = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)
