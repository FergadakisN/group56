# FastAPI Inference API

## Overview

The Fish Species Classification API provides REST endpoints for inference using trained ResNet models. It supports image uploads, returns predictions with confidence scores, and includes health check and model metadata endpoints.

## Starting the API

### Local Development

```bash
# Start the API server
uvicorn src.group_56.api:app --host 0.0.0.0 --port 8000 --reload

# Or using Python
python -m group_56.api
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### Using Docker

```bash
# Build the Docker image
docker build -t fish-api -f dockerfiles/api.dockerfile .

# Run the container
docker run -p 8000:8000 fish-api
```

## API Endpoints

### Root Endpoint

**GET /** - API information

```bash
curl http://localhost:8000/
```

Response:

```json
{
  "message": "Fish Species Classification API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict (POST)",
    "model_info": "/model/info"
  }
}
```

### Health Check

**GET /health** - Check API and model status

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Model Information

**GET /model/info** - Get model metadata

```bash
curl http://localhost:8000/model/info
```

Response:

```json
{
  "architecture": "resnet18",
  "num_classes": 30,
  "classes": ["anchovy", "barracuda", ...],
  "checkpoint_path": "models/best.pt"
}
```

### Prediction

**POST /predict** - Classify a fish image

```bash
# With curl
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/fish.jpg" \
  -F "top_k=5"

# With Python
import requests

with open("fish.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"top_k": 5}
    )
print(response.json())
```

Response:

```json
{
  "predicted_class": "tuna",
  "confidence": 0.9234,
  "top_k_predictions": [
    {"class": "tuna", "confidence": 0.9234},
    {"class": "salmon", "confidence": 0.0512},
    {"class": "cod", "confidence": 0.0123},
    {"class": "bass", "confidence": 0.0089},
    {"class": "trout", "confidence": 0.0042}
  ],
  "model_arch": "resnet18"
}
```

Parameters:

- `file`: Image file (JPEG, PNG) - **Required**
- `top_k`: Number of top predictions to return (default: 5)

### Load Model

**POST /model/load** - Manually load a checkpoint

```bash
curl -X POST "http://localhost:8000/model/load?checkpoint_path=models/best.pt"
```

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to test endpoints directly in your browser.

## Model Loading

The API automatically loads a model checkpoint on startup from these locations (in order):

1. `models/best.pt`
2. `outputs/resnet_run/best.pt`
3. `best.pt`

You can also load a custom checkpoint using the `/model/load` endpoint.

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad request (e.g., invalid image file)
- **500**: Server error (e.g., prediction failed)
- **503**: Service unavailable (e.g., model not loaded)

Example error response:

```json
{
  "detail": "Model not loaded"
}
```

## Testing

Run the API test suite:

```bash
# Run all API tests
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_predict_with_valid_image -v

# With coverage
pytest tests/test_api.py --cov=src/group_56/api
```

## Performance Considerations

- **Device Selection**: The API automatically uses CUDA if available, otherwise MPS (Apple Silicon) or CPU
- **Batch Size**: Currently processes one image at a time; can be extended for batch processing
- **Model Caching**: Model is loaded once at startup and kept in memory
- **Concurrent Requests**: FastAPI handles concurrent requests efficiently using async/await

## Example Client Script

```python
#!/usr/bin/env python
"""Example client for the Fish Classification API."""

import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def predict_fish(image_path: str, top_k: int = 3) -> dict:
    """Send an image to the API and get predictions."""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            params={"top_k": top_k}
        )
    
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <image_path>")
        sys.exit(1)
    
    result = predict_fish(sys.argv[1])
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.2%}")
```

## Deployment Notes

### Environment Variables

Set these environment variables for production:

```bash
export MODEL_PATH=models/best.pt  # Path to checkpoint
export API_HOST=0.0.0.0           # Bind address
export API_PORT=8000              # Port number
export LOG_LEVEL=INFO             # Logging level
```

### Production Server

For production, use a production ASGI server like Gunicorn:

```bash
gunicorn src.group_56.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Cloud Deployment

The API is ready for deployment to:

- **Google Cloud Run**: Use `dockerfiles/api.dockerfile`
- **AWS Lambda**: Add serverless wrapper (e.g., Mangum)
- **Azure Container Instances**: Use Docker image
- **Kubernetes**: Create deployment and service manifests

See Week 2 deployment tasks for GCP Cloud Run deployment instructions.
