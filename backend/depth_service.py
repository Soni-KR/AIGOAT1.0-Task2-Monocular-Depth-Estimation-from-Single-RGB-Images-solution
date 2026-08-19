import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import onnxruntime as ort
except ImportError:  # Keeps imports readable before dependencies are installed.
    ort = None


IMAGE_SIZE = 448
BATCH_SIZE = 8
MAX_SOURCE_PIXELS = 40_000_000
MODEL_PATH = Path(__file__).resolve().parents[1] / "depth_model.onnx"


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


class DepthModelService:
    """
    Owns the ONNX Runtime session and the image/depth conversion helpers.

    In a web backend, we usually keep model loading outside the route function.
    That way the model loads once when the server starts, not once per request.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.input_name = "input"
        self.output_name = "output"
        self.input_shape = [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
        self.output_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE]
        self._session = None

    @property
    def session(self):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install backend requirements first."
            )
        if self._session is None:
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            model_input = self._session.get_inputs()[0]
            model_output = self._session.get_outputs()[0]
            self.input_name = model_input.name
            self.output_name = model_output.name
            self.input_shape = list(model_input.shape)
            self.output_shape = list(model_output.shape)
        return self._session

    def ensure_ready(self) -> list[str]:
        """Load the model session and return its active execution providers."""
        return self.session.get_providers()

    def model_info(self) -> dict:
        # Accessing the session ensures the returned metadata came from the model.
        _ = self.session
        return {
            "model_name": self.model_path.name,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "image_size": IMAGE_SIZE,
        }

    def predict_from_image_bytes(self, image_bytes: bytes) -> dict:
        image_batch = self._preprocess_image(image_bytes)
        output = self.session.run([self.output_name], {self.input_name: image_batch})[0]
        depth = output[0].astype(np.float32)
        depth_norm = self._normalize_depth(depth)
        return {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "min_depth": float(depth.min()),
            "max_depth": float(depth.max()),
            "depth_png_base64": self._depth_to_base64_png(depth_norm),
        }

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as uploaded_image:
                if uploaded_image.width * uploaded_image.height > MAX_SOURCE_PIXELS:
                    raise InvalidImageError(
                        "The uploaded image dimensions are too large."
                    )
                uploaded_image.load()
                image = uploaded_image.convert("RGB")
        except InvalidImageError:
            raise
        except (
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise InvalidImageError(
                "The uploaded file is not a valid PNG or JPEG image."
            ) from exc

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

        array = np.array(image, dtype=np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        chw = np.transpose(array, (2, 0, 1))
        single = chw[np.newaxis, :, :, :]

        # The competition ONNX model has fixed batch size 8.
        return np.repeat(single, BATCH_SIZE, axis=0).astype(np.float32)

    def _normalize_depth(self, depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        low = float(depth.min())
        high = float(depth.max())
        return (depth - low) / (high - low + eps)

    def _depth_to_base64_png(self, depth_norm: np.ndarray) -> str:
        depth_uint8 = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(depth_uint8)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
