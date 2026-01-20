"""
FastAPI application for fish species classification.

This API provides endpoints for model inference, health checks, and model metadata.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from .data import get_official_transform
from .model import build_resnet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Load model
    try:
        # Try multiple common checkpoint locations
        checkpoint_paths = [
            Path("models/best.pt"),
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
    if MODEL is None or IDX_TO_CLASS is None or DEVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
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

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k_predictions=top_k_predictions,
            model_arch=arch,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
