from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from depth_service import DepthModelService
from schemas import HealthResponse, ModelInfoResponse, PredictionResponse


app = FastAPI(
    title="AIGOAT Task 2 Depth API",
    description="FastAPI backend for monocular depth estimation with depth_model.onnx.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

depth_service = DepthModelService()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return depth_service.model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image.",
        )

    image_bytes = await file.read()

    try:
        return depth_service.predict_from_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
