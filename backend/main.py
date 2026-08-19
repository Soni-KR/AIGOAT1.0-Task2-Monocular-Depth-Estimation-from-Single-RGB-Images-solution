import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import Settings
from backend.depth_service import DepthModelService, InvalidImageError
from backend.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
)


LOGGER = logging.getLogger(__name__)
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


async def read_upload_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload incrementally and stop as soon as it exceeds the limit."""
    chunks: list[bytes] = []
    total_bytes = 0

    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def create_app(
    settings: Settings | None = None,
    depth_service: DepthModelService | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime_depth_service = depth_service or DepthModelService()

    web_app = FastAPI(
        title="AIGOAT Task 2 Depth API",
        description="FastAPI backend for monocular depth estimation with depth_model.onnx.",
        version="0.2.0",
    )
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @web_app.get("/health", response_model=HealthResponse)
    def health():
        """Liveness: confirm that the API process can answer HTTP requests."""
        return {"status": "ok"}

    @web_app.get("/ready", response_model=ReadinessResponse)
    def ready():
        """Readiness: confirm that the ONNX model can serve predictions."""
        try:
            providers = runtime_depth_service.ensure_ready()
        except Exception as exc:
            LOGGER.exception("Model readiness check failed")
            raise HTTPException(
                status_code=503,
                detail="The inference model is not ready.",
            ) from exc

        return {"status": "ready", "providers": providers}

    @web_app.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        try:
            return runtime_depth_service.model_info()
        except Exception as exc:
            LOGGER.exception("Could not read model metadata")
            raise HTTPException(
                status_code=503,
                detail="Model metadata is unavailable.",
            ) from exc

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PNG or JPEG image.",
            )

        try:
            image_bytes = await read_upload_with_limit(
                file,
                runtime_settings.max_upload_bytes,
            )
        finally:
            await file.close()

        try:
            return runtime_depth_service.predict_from_image_bytes(image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Depth prediction failed")
            raise HTTPException(
                status_code=500,
                detail="Depth prediction failed.",
            ) from exc

    return web_app


app = create_app()
