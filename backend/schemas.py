from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    providers: list[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    input_name: str
    output_name: str
    input_shape: list[int | str | None]
    output_shape: list[int | str | None]
    image_size: int


class PredictionResponse(BaseModel):
    width: int
    height: int
    min_depth: float
    max_depth: float
    depth_png_base64: str
