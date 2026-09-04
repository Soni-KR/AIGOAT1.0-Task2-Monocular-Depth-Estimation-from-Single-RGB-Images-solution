# AIGOAT Task 2 - Monocular Depth Estimation

This repository contains our AIGOAT Task 2 solution and a learning-focused web application around the exported ONNX model.

The React frontend accepts a JPG or PNG image, sends it to FastAPI, and displays the predicted depth map. The FastAPI backend performs the same preprocessing used for the competition model and runs inference with ONNX Runtime.

## Project Structure

```text
task2_app/
|-- backend/
|   |-- main.py          FastAPI routes
|   |-- depth_service.py ONNX preprocessing and inference
|   `-- requirements.txt
|-- frontend/            Vite + React application
|-- depth_model.onnx     Exported competition model
|-- the one.ipynb        Original solution notebook
|-- task2.pdf            Competition task statement
|-- *.jpg                Sample input images
`-- README.md
```

## Requirements

- Python 3.10 or newer
- Node.js `^20.19.0` or `>=22.12.0`
- npm

## Backend Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Backend Docker

Build the production backend image from the repository root:

```powershell
docker build --file backend/Dockerfile --tag aigoat-api:0.2 .
```

Run the API with local frontend access and a 10 MB upload limit:

```powershell
docker run --rm --name aigoat-api `
    --publish 8000:8000 `
    --env AIGOAT_CORS_ORIGINS=http://localhost:5173 `
    --env AIGOAT_MAX_UPLOAD_MB=10 `
    aigoat-api:0.2
```

The container runs as a non-root Linux user and reports readiness through its
Docker health check. See
[`docs/phase-2-backend-container.md`](docs/phase-2-backend-container.md) for the
Dockerfile explanation, verification commands, and troubleshooting notes.

## Full Stack with Docker Compose

Build and start the production-shaped frontend and backend:

```powershell
docker compose up --build --detach
docker compose ps
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Nginx serves the compiled
React application and forwards `/api/*` to the private FastAPI service.

Inspect logs and stop the stack:

```powershell
docker compose logs --follow
docker compose down
```

See [`docs/phase-3-full-stack-compose.md`](docs/phase-3-full-stack-compose.md)
for the multi-stage frontend image, Nginx proxy, Compose network, health checks,
and complete verification workflow.

## Frontend Setup

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

The frontend uses `http://127.0.0.1:8000` by default. To use another backend address, copy `frontend/.env.example` to `frontend/.env` and change `VITE_API_BASE_URL`.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check whether the API process is alive |
| `GET` | `/ready` | Verify that ONNX Runtime can load the model |
| `GET` | `/model-info` | Inspect model input and output metadata |
| `POST` | `/predict` | Upload a JPG or PNG and receive a depth map |

`POST /predict` returns the output dimensions, minimum and maximum raw depth values, and the normalized depth image as a base64-encoded PNG.

## Verification

Install development dependencies and run the complete backend test suite:

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest
```

Run only the fast tests while developing:

```powershell
python -m pytest -m "not integration"
```

Frontend checks:

```powershell
cd frontend
npm run lint
npm run build
```

Backend smoke test after starting the API:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/ready -UseBasicParsing
```

## Backend Configuration

The backend reads deployment-specific settings from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AIGOAT_CORS_ORIGINS` | Local Vite origins | Comma-separated browser origins permitted by CORS |
| `AIGOAT_MAX_UPLOAD_MB` | `10` | Maximum uploaded image size in whole megabytes |

See [`backend/.env.example`](backend/.env.example) for example values. Environment
variables are read by the process; the application does not automatically load the
example file.

The implementation and reasoning for this production baseline are explained in
[`docs/phase-1-reliable-backend.md`](docs/phase-1-reliable-backend.md).

## Competition Files

The original notebook and task PDF are included for study and reproducibility. The ONNX model uses a fixed input batch shape of `(8, 3, 448, 448)`; the backend repeats one uploaded image across the batch and returns the first predicted depth map.
