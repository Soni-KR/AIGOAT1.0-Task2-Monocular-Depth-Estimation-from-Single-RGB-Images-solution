from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_path: str
    input_name: str
    output_name: str
    input_shape: list[int]
    output_shape: list[int]
    image_size: int


class PredictionResponse(BaseModel):
    width: int
    height: int
    min_depth: float
    max_depth: float
    depth_png_base64: str
