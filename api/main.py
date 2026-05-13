"""
Insurance Fraud Detection - FastAPI Production API
====================================================
Endpoints:
  GET  /health           -> Health check
  POST /predict           -> Predict fraud (default FP32)
  POST /predict-fp32      -> Predict fraud (explicit FP32)
  POST /predict-int8      -> Predict fraud (quantized INT8)
  GET  /metrics           -> Prometheus metrics
  GET  /model-info        -> Model metadata
  POST /benchmark         -> Run FP32 vs INT8 benchmark
"""

import os
import sys
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

from api.schemas import (
    PredictionRequest, PredictionResponse, HealthResponse,
    ModelInfoResponse, BenchmarkRequest, BenchmarkResponse,
)

# - Logging -
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fraud-api")

# - Prometheus Metrics -
REQUEST_COUNT = Counter(
    "fraud_api_request_count",
    "Total number of prediction requests",
    ["endpoint", "model_type", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "model_type"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
ERROR_COUNT = Counter(
    "fraud_api_error_count",
    "Total number of errors",
    ["endpoint", "error_type"],
)
MODEL_LOADED = Gauge(
    "fraud_api_model_loaded",
    "Whether the model is loaded (1) or not (0)",
    ["model_type"],
)
PREDICTION_DISTRIBUTION = Counter(
    "fraud_api_prediction_distribution",
    "Distribution of predictions",
    ["model_type", "prediction"],
)

# - Global Model State -
models = {
    "fp32": None,
    "int8": None,
}
model_metadata = {}

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_models():
    """Load models from local artifacts (preferred) or MLflow."""
    global models, model_metadata

    # ── 1. Try Local Loading (Fastest & Reliable) ─────────────────────
    fp32_local_path = os.path.join(PROJECT_ROOT, "artifacts", "model_fp32.joblib")
    if os.path.exists(fp32_local_path):
        try:
            models["fp32"] = joblib.load(fp32_local_path)
            logger.info("FP32 model loaded from local artifacts")
            MODEL_LOADED.labels(model_type="fp32").set(1)
        except Exception as e:
            logger.warning(f"Local FP32 load failed: {e}")

    # ── 2. Try MLflow (Fallback) ──────────────────────────────────────
    if models["fp32"] is None:
        logger.info("Trying to load FP32 model from MLflow registry...")
        try:
            import mlflow
            model_name = "insurance-fraud-detector"
            # Try to load via MLflow if flavor is available
            if hasattr(mlflow, "sklearn"):
                for uri in [f"models:/{model_name}@champion", f"models:/{model_name}/Production"]:
                    try:
                        models["fp32"] = mlflow.sklearn.load_model(uri)
                        logger.info(f"FP32 model loaded from MLflow: {uri}")
                        MODEL_LOADED.labels(model_type="fp32").set(1)
                        break
                    except Exception:
                        continue
            
            # If still None, try via Client + download
            if models["fp32"] is None:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest = sorted(versions, key=lambda v: int(v.version))[-1]
                    local_dir = client.download_artifacts(latest.run_id, "model")
                    models["fp32"] = joblib.load(os.path.join(local_dir, "model.pkl"))
                    logger.info(f"FP32 model downloaded and loaded: version {latest.version}")
                    MODEL_LOADED.labels(model_type="fp32").set(1)
        except Exception as e:
            logger.error(f"MLflow fallback failed: {e}")

    # - INT8 Model -
    int8_path = os.path.join(PROJECT_ROOT, "artifacts", "model_int8.joblib")
    if os.path.exists(int8_path):
        models["int8"] = joblib.load(int8_path)
        logger.info("INT8 quantized model loaded from artifacts/")
        MODEL_LOADED.labels(model_type="int8").set(1)
    else:
        logger.warning("INT8 model not found. Run `python -m api.quantize` first.")
        MODEL_LOADED.labels(model_type="int8").set(0)

    # - Metadata -
    if models["fp32"] is not None:
        fp32_size = _get_model_size(models["fp32"])
        model_metadata["fp32"] = {
            "type": type(models["fp32"]).__name__,
            "size_mb": round(fp32_size / (1024 * 1024), 2),
            "status": "loaded",
        }

    if models["int8"] is not None:
        int8_size = _get_model_size(models["int8"])
        model_metadata["int8"] = {
            "type": type(models["int8"]).__name__,
            "size_mb": round(int8_size / (1024 * 1024), 2),
            "status": "loaded",
        }
        if "fp32" in model_metadata:
            ratio = model_metadata["fp32"]["size_mb"] / max(model_metadata["int8"]["size_mb"], 0.01)
            model_metadata["compression_ratio"] = round(ratio, 2)


def _get_model_size(model) -> int:
    """Estimate model size in bytes using joblib serialization."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(model, f.name)
        size = os.path.getsize(f.name)
    os.unlink(f.name)
    return size


# - App Lifecycle -
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    logger.info("API ready - models loaded")
    yield
    logger.info("API shutting down")


# - FastAPI App -
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Production API for insurance fraud prediction with FP32/INT8 models",
    version="1.0.0",
    lifespan=lifespan,
)


# -
# FEATURE COLUMNS (must match training pipeline)
# -

FEATURE_COLS = [
    "age", "claim_amount", "policy_annual_premium",
    "months_as_customer", "num_previous_claims",
    "police_report_filed", "witnesses",
    "injury_claim", "vehicle_claim", "total_claim_amount",
    "insurance_type_encoded", "region_encoded",
]


def _prepare_features(req: PredictionRequest) -> pd.DataFrame:
    """Convert request to feature DataFrame."""
    data = {
        "age": req.age,
        "claim_amount": req.claim_amount,
        "policy_annual_premium": req.policy_annual_premium,
        "months_as_customer": req.months_as_customer,
        "num_previous_claims": req.num_previous_claims,
        "police_report_filed": req.police_report_filed,
        "witnesses": req.witnesses,
        "injury_claim": req.injury_claim,
        "vehicle_claim": req.vehicle_claim,
        "total_claim_amount": req.total_claim_amount,
        "insurance_type_encoded": req.insurance_type_encoded,
        "region_encoded": req.region_encoded,
    }
    return pd.DataFrame([data])[FEATURE_COLS]


def _predict(model, features: pd.DataFrame, model_type: str) -> PredictionResponse:
    """Run prediction and return response."""
    start = time.perf_counter()
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    fraud_probability = float(probabilities[1])
    confidence = float(max(probabilities))

    PREDICTION_DISTRIBUTION.labels(
        model_type=model_type,
        prediction="fraud" if prediction == 1 else "legit",
    ).inc()

    return PredictionResponse(
        prediction=prediction,
        label="FRAUD" if prediction == 1 else "LEGIT",
        fraud_probability=round(fraud_probability, 6),
        confidence=round(confidence, 6),
        model_type=model_type,
        latency_ms=round(latency_ms, 3),
    )


# -
# ENDPOINTS
# -

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        fp32_loaded=models["fp32"] is not None,
        int8_loaded=models["int8"] is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(req: PredictionRequest):
    """Predict fraud using the default (FP32) model."""
    return await predict_fp32(req)


@app.post("/predict-fp32", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fp32(req: PredictionRequest):
    """Predict fraud using the FP32 (full precision) model."""
    if models["fp32"] is None:
        ERROR_COUNT.labels(endpoint="/predict-fp32", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="FP32 model not loaded")

    start = time.perf_counter()
    try:
        features = _prepare_features(req)
        response = _predict(models["fp32"], features, "fp32")
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(endpoint="/predict-fp32", model_type="fp32", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict-fp32", model_type="fp32").observe(elapsed)
        return response
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict-fp32", error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(endpoint="/predict-fp32", model_type="fp32", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-int8", response_model=PredictionResponse, tags=["Prediction"])
async def predict_int8(req: PredictionRequest):
    """Predict fraud using the INT8 (quantized) model."""
    if models["int8"] is None:
        ERROR_COUNT.labels(endpoint="/predict-int8", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="INT8 model not loaded. Run `python -m api.quantize` first.")

    start = time.perf_counter()
    try:
        features = _prepare_features(req)
        response = _predict(models["int8"], features, "int8")
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(endpoint="/predict-int8", model_type="int8", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict-int8", model_type="int8").observe(elapsed)
        return response
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict-int8", error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(endpoint="/predict-int8", model_type="int8", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about loaded models."""
    return ModelInfoResponse(
        fp32=model_metadata.get("fp32"),
        int8=model_metadata.get("int8"),
        compression_ratio=model_metadata.get("compression_ratio"),
    )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
async def run_benchmark(req: BenchmarkRequest):
    """Run a quick FP32 vs INT8 benchmark."""
    from api.benchmark import run_comparison_benchmark

    try:
        results = run_comparison_benchmark(
            model_fp32=models["fp32"],
            model_int8=models["int8"],
            n_requests=req.n_requests,
        )
        return BenchmarkResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# - Middleware: request timing -
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header to all responses."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
