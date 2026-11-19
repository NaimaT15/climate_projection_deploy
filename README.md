# Climate Delta Inference Service

Stateless FastAPI service. Loads pre-trained model bundles into memory at startup and
serves climate-delta predictions. Inference only - does not retrain anything. Called by
the existing backend/UI; ships with no UI of its own.

## Required external artifact: model_bundles/

This service does NOT include the trained model bundles - they are ~1.2GB of binary
`.joblib` files and are delivered separately from this code package. At startup the
service reads them from the path in `MODEL_BUNDLES_DIR` (default: `/model_bundles`
inside the container). Mount or sync that directory before starting the service; it
must contain `manifest.json`, `reference_data/`, and one subdirectory per
`{GCM}_{Variable}_{Granularity}` bundle (each with `metadata.json`, `schema.json`,
`scaler.joblib`, `model_rf.joblib`, `model_xgboost.joblib`, `model_lightgbm.joblib`,
`model_ridge.joblib`).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BUNDLES_DIR` | `/model_bundles` (via Dockerfile) | Path to the externally-delivered model bundle directory |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins. Set this to the real calling backend/UI origin(s) in production instead of leaving it wide open |
| `PORT` | `8000` | Not read by the app itself - pass `--port` to uvicorn / map in the container runtime |

## Run with Docker

```
docker build -t climate-inference .
docker run -p 8000:8000 \
  -v /path/to/model_bundles:/model_bundles:ro \
  -e ALLOWED_ORIGINS=https://your-calling-backend.example.com \
  climate-inference
```

## Run without Docker

```
pip install -r requirements.txt
export MODEL_BUNDLES_DIR=/path/to/model_bundles
export ALLOWED_ORIGINS=https://your-calling-backend.example.com
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` - liveness check
- `GET /models` - lists available GCM x variable x granularity bundles
- `POST /predict` - one GCM, one model, one uploaded CSV/xlsx
- `POST /predict/compare_models` - one GCM, all models, one uploaded file
- `GET /predict/ensemble` - all GCMs' own reference data, individual + ensemble
- `POST /predict/upload_ensemble` - all GCMs run on one uploaded file, individual + ensemble

Full endpoint documentation is auto-generated at `/docs` (Swagger UI) once the service
is running.
