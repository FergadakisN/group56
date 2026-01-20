"""
Unit tests for the FastAPI application.

Tests cover health endpoints, model info, and prediction endpoints.
"""

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from group_56.api import app, load_model


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a mock checkpoint file."""
    checkpoint_path = tmp_path / "test_checkpoint.pt"

    # Create a minimal checkpoint
    class_to_idx = {"fish_a": 0, "fish_b": 1, "fish_c": 2}
    num_classes = len(class_to_idx)

    # Create a dummy model state dict
    from group_56.model import build_resnet

    model = build_resnet(num_classes=num_classes, arch="resnet18", pretrained=False)

    checkpoint = {
        "arch": "resnet18",
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "model_state_dict": model.state_dict(),
        "epoch": 1,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint(client):
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_endpoint_no_model(client):
    """Test health endpoint when no model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_health_endpoint_with_model(client, mock_checkpoint):
    """Test health endpoint after loading a model."""
    load_model(mock_checkpoint)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "device" in data


def test_model_info_without_model(client):
    """Test model info endpoint fails when model not loaded."""
    # Reset model state
    from group_56 import api

    api.MODEL = None
    api.CLASS_TO_IDX = None

    response = client.get("/model/info")
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_model_info_with_model(client, mock_checkpoint):
    """Test model info endpoint returns metadata."""
    load_model(mock_checkpoint)
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["architecture"] == "resnet18"
    assert data["num_classes"] == 3
    assert len(data["classes"]) == 3
    assert "fish_a" in data["classes"]


def test_predict_without_model(client, sample_image):
    """Test prediction fails when model not loaded."""
    from group_56 import api

    api.MODEL = None

    response = client.post("/predict", files={"file": ("test.jpg", sample_image, "image/jpeg")})
    assert response.status_code == 503


def test_predict_with_invalid_file(client, mock_checkpoint):
    """Test prediction with non-image file."""
    load_model(mock_checkpoint)

    # Create a text file instead of image
    text_file = io.BytesIO(b"not an image")
    response = client.post("/predict", files={"file": ("test.txt", text_file, "text/plain")})
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_predict_with_valid_image(client, mock_checkpoint, sample_image):
    """Test successful prediction with valid image."""
    load_model(mock_checkpoint)

    response = client.post("/predict", files={"file": ("test.jpg", sample_image, "image/jpeg")})
    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_k_predictions" in data
    assert data["predicted_class"] in ["fish_a", "fish_b", "fish_c"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_k_predictions"]) <= 5


def test_predict_top_k_parameter(client, mock_checkpoint, sample_image):
    """Test prediction with custom top_k parameter."""
    load_model(mock_checkpoint)

    response = client.post("/predict?top_k=2", files={"file": ("test.jpg", sample_image, "image/jpeg")})
    assert response.status_code == 200

    data = response.json()
    assert len(data["top_k_predictions"]) == 2


def test_load_model_endpoint(client, mock_checkpoint):
    """Test manual model loading endpoint."""
    response = client.post(f"/model/load?checkpoint_path={mock_checkpoint}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_info" in data


def test_load_model_invalid_path(client):
    """Test loading model with invalid path."""
    response = client.post("/model/load?checkpoint_path=nonexistent.pt")
    assert response.status_code == 500


def test_multiple_predictions(client, mock_checkpoint):
    """Test multiple consecutive predictions."""
    load_model(mock_checkpoint)

    for i in range(3):
        img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post("/predict", files={"file": (f"test_{i}.jpg", img_bytes, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data


def test_prediction_consistency(client, mock_checkpoint, sample_image):
    """Test that same image produces consistent predictions."""
    load_model(mock_checkpoint)

    # Make two predictions with the same image
    img_bytes_1 = sample_image
    img_bytes_1.seek(0)

    img_bytes_2 = io.BytesIO(img_bytes_1.getvalue())

    response1 = client.post("/predict", files={"file": ("test.jpg", img_bytes_1, "image/jpeg")})
    response2 = client.post("/predict", files={"file": ("test.jpg", img_bytes_2, "image/jpeg")})

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    assert data1["predicted_class"] == data2["predicted_class"]
    assert abs(data1["confidence"] - data2["confidence"]) < 1e-5
