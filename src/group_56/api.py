"""
FastAPI application for fish species classification.

This API provides endpoints for model inference, health checks, and model metadata.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from google.cloud import storage
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY, CollectorRegistry
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.asgi import make_asgi_app

from .data import get_official_transform
from .model import build_resnet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
request_count = Counter(
    "fish_api_requests_total",
    "Total number of requests to the API",
    ["method", "endpoint"],
)
error_count = Counter(
    "fish_api_errors_total",
    "Total number of errors in the API",
    ["method", "endpoint", "error_type"],
)
prediction_latency = Histogram(
    "fish_api_prediction_latency_seconds",
    "Time taken to make predictions in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
)
model_load_time = Histogram(
    "fish_api_model_load_time_seconds",
    "Time taken to load the model in seconds",
)

# Global model and metadata
MODEL: nn.Module | None = None
CLASS_TO_IDX: dict[str, int] | None = None
IDX_TO_CLASS: dict[int, str] | None = None
DEVICE: torch.device | None = None
MODEL_INFO: dict[str, Any] = {}


class TopKPrediction(BaseModel):
    """Item model for top-k predictions."""

    class_name: str = Field(alias="class")
    confidence: float

    model_config = {
        "populate_by_name": True,
    }


class PredictionResponse(BaseModel):
    """Response model for classification predictions."""

    predicted_class: str
    confidence: float
    top_k_predictions: list[TopKPrediction]
    model_arch: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Response model for model metadata."""

    architecture: str
    num_classes: int
    classes: list[str]
    checkpoint_path: str | None


def load_model(checkpoint_path: str | Path = "models/best.pt") -> None:
    """
    Load the model checkpoint into memory.

    Args:
        checkpoint_path: Path to the model checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If checkpoint is malformed.
    """
    global MODEL, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE, MODEL_INFO

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")

    # Determine device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    logger.info(f"Using device: {DEVICE}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Extract metadata
    arch = checkpoint.get("arch", "resnet18")
    num_classes = checkpoint.get("num_classes")
    CLASS_TO_IDX = checkpoint.get("class_to_idx", {})

    if num_classes is None or not CLASS_TO_IDX:
        raise RuntimeError("Checkpoint missing 'num_classes' or 'class_to_idx'")

    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

    # Build model
    MODEL = build_resnet(num_classes=num_classes, arch=arch, pretrained=False)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()

    MODEL_INFO = {
        "architecture": arch,
        "num_classes": num_classes,
        "checkpoint_path": str(checkpoint_path),
        "device": str(DEVICE),
    }

    logger.info(f"Model loaded: {arch} with {num_classes} classes on {DEVICE}")


def download_model_from_gcs(bucket: str, object_path: str, local_path: str | Path) -> bool:
    """Download model checkpoint from Google Cloud Storage using the Python client."""

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Attempting to download model gs://{bucket}/{object_path}")
        client = storage.Client()
        blob = client.bucket(bucket).blob(object_path)
        blob.download_to_filename(local_path)
        logger.info(f"Successfully downloaded model to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model via google-cloud-storage: {e}")
        return False


def download_model_with_gsutil(bucket_path: str, local_path: str | Path) -> bool:
    """Fallback downloader using gsutil if installed."""

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["gsutil", "cp", bucket_path, str(local_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Successfully downloaded model to {local_path} using gsutil")
            return True
        logger.error(f"Failed to download model with gsutil: {result.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error downloading model with gsutil: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Load model with timing metrics
    start_time = time.time()
    try:
        # Try to download from GCS if BUCKET_NAME env var is set
        gcs_bucket = os.getenv("GCS_BUCKET", "fish_mlops")
        gcs_object = os.getenv("GCS_MODEL_OBJECT", "models/fish_classifier.pt")
        gcs_model_path = f"gs://{gcs_bucket}/{gcs_object}"
        local_model_path = Path("/tmp/fish_classifier.pt")

        downloaded = download_model_from_gcs(gcs_bucket, gcs_object, local_model_path)
        if not downloaded:
            logger.warning("google-cloud-storage download failed, attempting gsutil fallback")
            downloaded = download_model_with_gsutil(gcs_model_path, local_model_path)

        if downloaded:
            load_model(local_model_path)
        else:
            checkpoint_paths = [
                Path("models/best.pt"),
                Path("models/quick_deploy/fish_classifier.pt"),
                Path("outputs/resnet_run/best.pt"),
                Path("best.pt"),
            ]

            for ckpt_path in checkpoint_paths:
                if ckpt_path.exists():
                    load_model(ckpt_path)
                    break
            else:
                logger.warning("No model checkpoint found at startup. API will run without loaded model.")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")

    yield

    # Shutdown: cleanup if needed
    logger.info("API shutting down")


app = FastAPI(
    title="Fish Species Classification API",
    description="Deep learning inference API for identifying fish species from images",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class RootResponse(BaseModel):
    """Response model for root endpoint."""

    message: str
    version: str
    endpoints: dict[str, str]


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint with API information."""
    return RootResponse(
        message="Fish Species Classification API",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "predict": "/predict (POST)",
            "model_info": "/model/info",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE) if DEVICE else "none",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Get model metadata."""
    if MODEL is None or CLASS_TO_IDX is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        architecture=MODEL_INFO.get("architecture", "unknown"),
        num_classes=MODEL_INFO.get("num_classes", 0),
        classes=sorted(CLASS_TO_IDX.keys()),
        checkpoint_path=MODEL_INFO.get("checkpoint_path"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file for classification"),  # noqa: B008
    top_k: int = 5,
) -> PredictionResponse:
    """
    Predict fish species from an uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG).
        top_k: Number of top predictions to return.

    Returns:
        PredictionResponse with predicted class and confidence scores.

    Raises:
        HTTPException: If model not loaded or invalid image.
    """
    request_count.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    if MODEL is None or IDX_TO_CLASS is None or DEVICE is None:
        error_count.labels(method="POST", endpoint="/predict", error_type="ModelNotLoaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        error_count.labels(method="POST", endpoint="/predict", error_type="InvalidFileType").inc()
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Get the appropriate transform for the model architecture
        arch = MODEL_INFO.get("architecture", "resnet18")
        transform = get_official_transform(arch)

        # Transform and add batch dimension
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = MODEL(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(IDX_TO_CLASS)))

        top_k_predictions = [
            TopKPrediction(class_name=IDX_TO_CLASS[idx.item()], confidence=prob.item())
            for prob, idx in zip(top_k_probs, top_k_indices, strict=True)
        ]

        # Get top prediction
        predicted_idx = top_k_indices[0].item()
        predicted_class = IDX_TO_CLASS[predicted_idx]
        confidence = top_k_probs[0].item()

        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        prediction_latency.observe(time.time() - start_time)

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k_predictions=top_k_predictions,
            model_arch=arch,
        )

    except HTTPException:
        prediction_latency.observe(time.time() - start_time)
        raise
    except Exception as e:
        error_count.labels(method="POST", endpoint="/predict", error_type="PredictionError").inc()
        logger.error(f"Prediction error: {e}")
        prediction_latency.observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@app.post("/model/load")
async def load_model_endpoint(checkpoint_path: str) -> JSONResponse:
    """
    Manually load a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        JSON response with loading status.
    """
    try:
        load_model(checkpoint_path)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model loaded from {checkpoint_path}",
                "model_info": MODEL_INFO,
            }
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
