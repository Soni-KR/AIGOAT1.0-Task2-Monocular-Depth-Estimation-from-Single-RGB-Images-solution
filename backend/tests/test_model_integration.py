import base64
from pathlib import Path

import pytest

from backend.depth_service import IMAGE_SIZE, DepthModelService


ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = ROOT_DIR / "a0261933-46ac-404b-b8b1-d0486b411615.jpg"


@pytest.mark.integration
def test_real_model_predicts_a_depth_png():
    service = DepthModelService()

    result = service.predict_from_image_bytes(SAMPLE_IMAGE.read_bytes())

    assert result["width"] == IMAGE_SIZE
    assert result["height"] == IMAGE_SIZE
    assert result["min_depth"] <= result["max_depth"]
    assert base64.b64decode(result["depth_png_base64"]).startswith(b"\x89PNG")
