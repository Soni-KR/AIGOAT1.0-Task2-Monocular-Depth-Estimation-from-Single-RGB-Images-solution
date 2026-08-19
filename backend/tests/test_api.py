from fastapi.testclient import TestClient

from backend.config import Settings
from backend.depth_service import InvalidImageError
from backend.main import create_app


PREDICTION = {
    "width": 448,
    "height": 448,
    "min_depth": 0.1,
    "max_depth": 0.9,
    "depth_png_base64": "ZmFrZS1wbmc=",
}


class FakeDepthService:
    def __init__(self):
        self.ready_calls = 0
        self.predict_calls = 0
        self.readiness_error = None

    def ensure_ready(self):
        self.ready_calls += 1
        if self.readiness_error:
            raise self.readiness_error
        return ["CPUExecutionProvider"]

    def model_info(self):
        return {
            "model_name": "depth_model.onnx",
            "input_name": "input",
            "output_name": "output",
            "input_shape": [8, 3, 448, 448],
            "output_shape": [8, 448, 448],
            "image_size": 448,
        }

    def predict_from_image_bytes(self, image_bytes):
        self.predict_calls += 1
        if image_bytes == b"invalid":
            raise InvalidImageError("The uploaded image is invalid.")
        if image_bytes == b"explode":
            raise RuntimeError("secret internal failure")
        return PREDICTION


def make_client(max_upload_mb=1):
    service = FakeDepthService()
    settings = Settings(
        cors_origins=("https://frontend.example.com",),
        max_upload_bytes=max_upload_mb * 1024 * 1024,
    )
    return TestClient(create_app(settings=settings, depth_service=service)), service


def test_health_is_a_cheap_liveness_check():
    client, service = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.ready_calls == 0


def test_ready_loads_the_model_service():
    client, service = make_client()

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "providers": ["CPUExecutionProvider"],
    }
    assert service.ready_calls == 1


def test_ready_returns_a_safe_503_when_model_loading_fails():
    client, service = make_client()
    service.readiness_error = RuntimeError("secret model path")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "The inference model is not ready."}
    assert "secret model path" not in response.text


def test_model_info_returns_model_metadata():
    client, _ = make_client()

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["input_shape"] == [8, 3, 448, 448]


def test_predict_rejects_unsupported_content_type_before_inference():
    client, service = make_client()

    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert service.predict_calls == 0


def test_predict_rejects_oversized_upload_before_inference():
    client, service = make_client(max_upload_mb=1)

    response = client.post(
        "/predict",
        files={"file": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Image exceeds the 1 MB limit."}
    assert service.predict_calls == 0


def test_predict_returns_invalid_image_as_a_client_error():
    client, _ = make_client()

    response = client.post(
        "/predict",
        files={"file": ("invalid.png", b"invalid", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The uploaded image is invalid."}


def test_predict_hides_internal_exception_details():
    client, _ = make_client()

    response = client.post(
        "/predict",
        files={"file": ("image.png", b"explode", "image/png")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Depth prediction failed."}
    assert "secret internal failure" not in response.text


def test_predict_returns_the_service_result():
    client, service = make_client()

    response = client.post(
        "/predict",
        files={"file": ("image.png", b"valid", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == PREDICTION
    assert service.predict_calls == 1


def test_cors_allows_only_the_configured_frontend_origin():
    client, _ = make_client()

    response = client.options(
        "/predict",
        headers={
            "Origin": "https://frontend.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://frontend.example.com"
    )
