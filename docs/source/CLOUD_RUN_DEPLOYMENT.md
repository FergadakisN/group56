# Cloud Run Deployment Guide

This guide covers deploying the Fish Species Classification API to Google Cloud Run.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed locally
- A trained model checkpoint

## Setup

### 1. Configure GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export SERVICE_NAME="fish-classifier-api"

# Authenticate and set project
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Build and Push Docker Image

```bash
# Build the image
docker build -f dockerfiles/api.dockerfile -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest
```

Alternatively, use Cloud Build:

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest -f dockerfiles/api.dockerfile .
```

### 3. Deploy to Cloud Run

#### Option A: Deploy without model (health checks only)

```bash
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --port 8080
```

#### Option B: Deploy with model from GCS

First, upload your trained model to Google Cloud Storage:

```bash
# Create a bucket
gsutil mb gs://$PROJECT_ID-models

# Upload your model
gsutil cp models/best.pt gs://$PROJECT_ID-models/fish-classifier/best.pt
```

Deploy with model mounting:

```bash
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --port 8080 \
  --set-env-vars MODEL_PATH=/models/best.pt \
  --execution-environment gen2
```

For production, use Cloud Run's volume mounts (requires GCS FUSE):

```bash
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --port 8080 \
  --add-volume name=models,type=cloud-storage,bucket=$PROJECT_ID-models \
  --add-volume-mount volume=models,mount-path=/app/models \
  --execution-environment gen2
```

### 4. Test Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test prediction (with image file)
curl -X POST "$SERVICE_URL/predict" \
  -F "file=@path/to/test_image.jpg"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `PORT` | API server port | `8080` |
| `MODEL_PATH` | Path to model checkpoint | `models/best.pt` |

### Resource Limits

Recommended settings based on traffic:

- **Low traffic**: 1 CPU, 1Gi memory, max 3 instances
- **Medium traffic**: 2 CPU, 2Gi memory, max 10 instances
- **High traffic**: 4 CPU, 4Gi memory, max 50 instances

## CI/CD with GitHub Actions

See [`.github/workflows/deploy-cloud-run.yaml`](../../.github/workflows/deploy-cloud-run.yaml) for automated deployment workflow.

The workflow:

1. Builds Docker image
2. Pushes to GCR
3. Deploys to Cloud Run
4. Runs smoke tests

## Monitoring

### View Logs

```bash
# Stream logs
gcloud run services logs tail $SERVICE_NAME --region $REGION

# View logs in Cloud Console
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

### Metrics

Monitor in Cloud Console:

- Request count and latency
- Error rate
- Memory and CPU utilization
- Cold start frequency

## Cost Optimization

1. **Set minimum instances to 0** for development (cold starts acceptable)
2. **Use concurrency**: Set `--concurrency 80` to handle multiple requests per instance
3. **Right-size resources**: Start small and scale based on metrics
4. **Enable CPU throttling**: Saves cost when idle (default behavior)

```bash
# Cost-optimized deployment
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --region $REGION \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 80 \
  --cpu-throttling
```

## Troubleshooting

### Common Issues

**Container failed to start**:

- Check logs: `gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50`
- Verify PORT environment variable is set to 8080
- Ensure uvicorn binds to `0.0.0.0`, not `localhost`

**Out of memory errors**:

- Increase memory limit: `--memory 4Gi`
- Check model size and batch processing

**Cold starts too slow**:

- Set `--min-instances 1` to keep one instance warm
- Optimize model loading in startup

**Authentication errors**:

- For public API: `--allow-unauthenticated`
- For private: Use Cloud IAM and require authentication

## Security

### Production Checklist

- [ ] Remove `--allow-unauthenticated` and use IAM
- [ ] Enable HTTPS (automatic with Cloud Run)
- [ ] Use Secret Manager for sensitive config
- [ ] Set up VPC connector for private resources
- [ ] Enable Cloud Armor for DDoS protection
- [ ] Implement rate limiting in API code
- [ ] Use least-privilege service accounts

### Using Secrets

```bash
# Create secret for model path or API keys
echo -n "gs://bucket/model.pt" | gcloud secrets create model-path --data-file=-

# Deploy with secret
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --update-secrets MODEL_PATH=model-path:latest
```

## Cleanup

```bash
# Delete the service
gcloud run services delete $SERVICE_NAME --region $REGION

# Delete the image
gcloud container images delete gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Delete the storage bucket
gsutil rm -r gs://$PROJECT_ID-models
```
