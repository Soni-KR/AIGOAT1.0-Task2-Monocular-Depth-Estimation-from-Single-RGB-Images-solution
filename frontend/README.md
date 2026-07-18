# React Frontend

This Vite + React application uploads JPG or PNG files to the FastAPI backend and displays the returned depth map.

```powershell
npm install
npm run dev
```

The development server runs at [http://127.0.0.1:5173](http://127.0.0.1:5173).

Configuration is optional. Copy `.env.example` to `.env` to change the backend URL:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Useful checks:

```powershell
npm run lint
npm run build
```

See the [project README](../README.md) for the full setup.
