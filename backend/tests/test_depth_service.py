import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.depth_service import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MAX_SOURCE_PIXELS,
    DepthModelService,
    InvalidImageError,
)


def make_png_bytes(color=(255, 128, 0)):
    image = Image.new("RGB", (20, 10), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_produces_the_fixed_model_batch():
    service = DepthModelService()

    batch = service._preprocess_image(make_png_bytes())

    assert batch.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() <= batch.max() <= 1.0
    np.testing.assert_array_equal(batch[0], batch[-1])


def test_preprocessing_rejects_fake_image_bytes():
    service = DepthModelService()

    with pytest.raises(InvalidImageError, match="valid PNG or JPEG"):
        service._preprocess_image(b"not an image")


def test_preprocessing_rejects_excessive_decoded_dimensions(monkeypatch):
    service = DepthModelService()

    class OversizedImage:
        width = MAX_SOURCE_PIXELS + 1
        height = 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(Image, "open", lambda *_: OversizedImage())

    with pytest.raises(InvalidImageError, match="dimensions are too large"):
        service._preprocess_image(b"compressed image bytes")


def test_normalizing_constant_depth_is_finite_and_zero():
    service = DepthModelService()

    normalized = service._normalize_depth(np.full((2, 2), 4.2, dtype=np.float32))

    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, np.zeros((2, 2), dtype=np.float32))


def test_depth_png_encoding_creates_a_grayscale_png():
    service = DepthModelService()
    depth = np.array([[0.0, 1.0]], dtype=np.float32)

    encoded = service._depth_to_base64_png(depth)
    decoded = base64.b64decode(encoded)

    with Image.open(BytesIO(decoded)) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        assert image.size == (2, 1)
