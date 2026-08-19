# FastAPI Backend

The backend exposes the ONNX depth model through three routes:

- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /predict`

Run these commands from the `task2_app` repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The model is loaded lazily from `task2_app/depth_model.onnx` when the first prediction is requested.

For development and tests, install `backend/requirements-dev.txt` instead. It
includes the runtime requirements plus pytest and HTTPX2:

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest
```

See the [project README](../README.md) for the complete setup and API behavior.
