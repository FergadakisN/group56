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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage  # type: ignore[import-untyped]
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_client.asgi import make_asgi_app
from pydantic import BaseModel, Field

from .data import get_official_transform
from .extract_features import extract_image_features, features_to_csv_row, get_csv_header
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
PREDICTION_DATABASE_PATH = Path("prediction_database.csv")


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


def log_prediction_to_csv(image: Image.Image, predicted_class: str) -> None:
    """
    Log prediction data to CSV file for drift detection.

    This function extracts image features and appends them along with the
    prediction and timestamp to a CSV database. This data can later be used
    to detect data drift.

    Args:
        image: PIL Image object that was classified
        predicted_class: The predicted class name
    """
    try:
        # Extract features from image
        features = extract_image_features(image)

        # Get current timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create CSV row
        csv_row = features_to_csv_row(features, timestamp, predicted_class)

        # Write to file (create with header if doesn't exist)
        if not PREDICTION_DATABASE_PATH.exists():
            with open(PREDICTION_DATABASE_PATH, "w") as f:
                f.write(get_csv_header() + "\n")
                f.write(csv_row + "\n")
        else:
            with open(PREDICTION_DATABASE_PATH, "a") as f:
                f.write(csv_row + "\n")

        logger.info(f"Logged prediction to {PREDICTION_DATABASE_PATH}")

    except Exception as e:
        # Don't fail prediction if logging fails
        logger.error(f"Failed to log prediction: {e}")


def save_to_gcs(local_path: Path, bucket_name: str, object_name: str) -> None:
    """
    Upload local file to GCS bucket as backup.

    Args:
        local_path: Local file path
        bucket_name: GCS bucket name
        object_name: Object path in bucket
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{object_name}")
    except Exception as e:
        logger.warning(f"Failed to backup prediction database to GCS: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Load model with timing metrics
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
            "monitoring": "/monitoring",
            "metrics": "/metrics",
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file for classification"),  # noqa: B008
    top_k: int = 5,
) -> PredictionResponse:
    """
    Predict fish species from an uploaded image.

    Args:
        background_tasks: FastAPI background tasks for async logging
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

        # Log prediction as background task (non-blocking)
        background_tasks.add_task(log_prediction_to_csv, image, predicted_class)

        # Optionally backup to GCS every 10 predictions
        if PREDICTION_DATABASE_PATH.exists():
            with open(PREDICTION_DATABASE_PATH) as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            if line_count % 10 == 0 and line_count > 0:
                gcs_bucket = os.getenv("GCS_BUCKET", "fish_mlops")
                background_tasks.add_task(
                    save_to_gcs,
                    PREDICTION_DATABASE_PATH,
                    gcs_bucket,
                    "monitoring/prediction_database.csv",
                )

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


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_endpoint(n_latest: int = 100):
    """
    Generate data drift monitoring report.

    This endpoint analyzes the logged predictions and generates an HTML report
    showing data drift, data quality, and target drift metrics.

    Args:
        n_latest: Number of latest predictions to analyze (default: 100)

    Returns:
        HTML report from Evidently
    """
    from fastapi.responses import HTMLResponse

    from .data_drift import generate_drift_report, load_current_data

    try:
        # Check if prediction database exists
        if not PREDICTION_DATABASE_PATH.exists():
            return HTMLResponse(
                content="""
                <html>
                <head><title>Monitoring - No Data</title></head>
                <body>
                    <h1>No Prediction Data Available</h1>
                    <p>The prediction database is empty. Make some predictions first:</p>
                    <pre>curl -X POST http://localhost:8000/predict -F file=@image.jpg</pre>
                </body>
                </html>
                """,
                status_code=200,
            )

        # Load data
        logger.info(f"Loading prediction data (latest {n_latest} entries)")
        current_data = load_current_data(str(PREDICTION_DATABASE_PATH), n_latest=n_latest)

        if len(current_data) < 10:
            return HTMLResponse(
                content=f"""
                <html>
                <head><title>Monitoring - Insufficient Data</title></head>
                <body>
                    <h1>Insufficient Data for Drift Detection</h1>
                    <p>Only {len(current_data)} predictions logged. Need at least 10 for meaningful analysis.</p>
                    <p>Make more predictions to enable drift detection.</p>
                </body>
                </html>
                """,
                status_code=200,
            )

        # For reference data, use the current data split in half
        # (first half as reference, second half as current)
        # In production, you'd load actual training data features
        split_idx = len(current_data) // 2
        reference_data = current_data.iloc[:split_idx].copy()
        current_data_subset = current_data.iloc[split_idx:].copy()

        logger.info(f"Running drift detection: {len(reference_data)} ref, {len(current_data_subset)} current")

        # Generate report in memory
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            report_path = tmp.name

        generate_drift_report(reference_data, current_data_subset, output_path=report_path)

        # Read and return HTML
        with open(report_path) as f:
            html_content = f.read()

        # Cleanup
        Path(report_path).unlink()

        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        logger.error(f"Monitoring endpoint error: {e}")
        return HTMLResponse(
            content=f"""
            <html>
            <head><title>Monitoring Error</title></head>
            <body>
                <h1>Error Generating Monitoring Report</h1>
                <p>Error: {str(e)}</p>
            </body>
            </html>
            """,
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
