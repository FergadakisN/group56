# Cloud Run Deployment Guide (M23)

This guide documents the complete process for deploying the Fish Species Classification API to Google Cloud Run with automatic model loading from Google Cloud Storage. This is based on the M23 milestone completion with verified working deployment pattern.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed locally
- A trained model checkpoint uploaded to GCS
- Artifact Registry repository created

## Quick Start (Recommended)

If you already have a trained model, deploy in 5 minutes:

```bash
export PROJECT_ID="fish-mlops"
export REGION="europe-west-1"
export SERVICE_NAME="fish-classifier-api"
export ARTIFACT_REPO="cloud-run-source-deploy"
export GCS_BUCKET="fish_mlops"

# 1. Upload model to GCS
gsutil cp models/fish_classifier.pt gs://$GCS_BUCKET/models/fish_classifier.pt

# 2. Build and push Docker image
docker build -t ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  -f dockerfiles/api.dockerfile .
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1

# 3. Deploy to Cloud Run with GCS model loading
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8080 \
  --timeout 300 \
  --set-env-vars GCS_BUCKET=$GCS_BUCKET,GCS_MODEL_OBJECT=models/fish_classifier.pt

# 4. Verify (should show model_loaded: true)
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')
curl -s $SERVICE_URL/health | jq .
```

---

## Detailed Setup Guide

### Step 1: Configure GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west-1"
export SERVICE_NAME="fish-classifier-api"

# Authenticate and set project
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### Step 2: Prepare Model for Deployment

#### Train and Save Model

```bash
# Train the model (or use existing checkpoint)
python src/group_56/train.py

# Verify model exists and size
ls -lh models/fish_classifier.pt
```

#### Upload Model to GCS

The API loads the model from GCS on startup using the `google-cloud-storage` client library:

```bash
# Create GCS bucket if it doesn't exist
gsutil mb -l $REGION gs://$GCS_BUCKET 2>/dev/null || echo "Bucket exists"

# Upload model checkpoint
gsutil cp models/fish_classifier.pt gs://$GCS_BUCKET/models/fish_classifier.pt

# Verify upload
gsutil ls -lh gs://$GCS_BUCKET/models/

# (Optional) Grant Cloud Run service account access
export RUN_SA="${PROJECT_ID}@appspot.gserviceaccount.com"
gsutil iam ch serviceAccount:$RUN_SA:objectViewer gs://$GCS_BUCKET
```

### Step 3: Build and Push Docker Image

The Docker image must include the `google-cloud-storage` dependency to download models from GCS.

**Using Artifact Registry (recommended):**

```bash
# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build the Docker image with Artifact Registry tag
docker build \
  -t ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  -f dockerfiles/api.dockerfile .

# Verify image was built
docker images | grep $SERVICE_NAME

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1

# List pushed images
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO
```

**Alternative: Use Cloud Build (no local Docker required):**

```bash
gcloud builds submit \
  --region=$REGION \
  --tag ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  -f dockerfiles/api.dockerfile .
```

### Step 4: Deploy to Cloud Run

#### Option A: Deploy with GCS Model Loading (Recommended for M23)

The API automatically downloads the model from GCS on startup using environment variables:

```bash
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8080 \
  --timeout 300 \
  --set-env-vars \
    GCS_BUCKET=$GCS_BUCKET,\
    GCS_MODEL_OBJECT=models/fish_classifier.pt

# View deployment details
gcloud run services describe $SERVICE_NAME --region $REGION
```

#### Option B: Deploy without Model (Testing Only)

For testing the API without loading a model:

```bash
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --port 8080
```

### Step 5: Verify Deployment

```bash
# Get the service URL
export SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')
echo "Service URL: $SERVICE_URL"

# Check health endpoint (should show model_loaded: true)
curl -s $SERVICE_URL/health | jq .

# Expected output (with model):
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cpu"
# }

# Test prediction (requires test image)
curl -X POST "$SERVICE_URL/predict" \
  -F "file=@path/to/test_image.jpg"

# View API documentation
open "$SERVICE_URL/docs"

# Stream logs in real-time
gcloud run logs read $SERVICE_NAME --region $REGION --limit 50 --follow
```

## M23 Completion Results

### Verified Working Deployment

- **Service URL**: [https://fish-classifier-api-170418683866.europe-west-1.run.app](https://fish-classifier-api-170418683866.europe-west-1.run.app)
- **Image**: europe-west-1-docker.pkg.dev/fish-mlops/cloud-run-source-deploy/fish-classifier-api:v3
- **Model Source**: gs://fish_mlops/models/fish_classifier.pt (43.6 MB)
- **Status**: ✅ Healthy, model_loaded: true on CPU

### Key Implementation Details

1. **Model Loading**: The API uses `google-cloud-storage` client library (not gsutil CLI) for reliable startup model loading
2. **Environment Variables**: Set `GCS_BUCKET` and `GCS_MODEL_OBJECT` at deployment time
3. **Fallback Logic**: If GCS download fails, API logs error and continues (health endpoint still responds)
4. **Dependency Resolution**: Use `fsspec==2024.2.0` for compatibility with `dvc-gs` and `gcsfs`

### Health Check Response

```bash
$ curl -s https://fish-classifier-api-170418683866.europe-west-1.run.app/health | jq .
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Configuration

### Environment Variables

| Variable           | Description                                              | Required | Default          |
| ------------------ | -------------------------------------------------------- | -------- | ---------------- |
| `GCS_BUCKET`       | GCS bucket name for model storage                        | No       | -                |
| `GCS_MODEL_OBJECT` | Path to model in GCS (e.g., `models/fish_classifier.pt`) | No       | -                |
| `PORT`             | API server port                                          | No       | `8080`           |
| `MODEL_PATH`       | Local fallback path to model                             | No       | `models/best.pt` |

### Resource Configuration

Recommended settings by traffic level:

- **Development**: 1 CPU, 1Gi memory, max 3 instances
- **Production (M23 tested)**: 2 CPU, 2Gi memory, max 10 instances
- **High-traffic**: 4 CPU, 4Gi memory, max 50 instances

```bash
# Example: Increase resources for high traffic
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --memory 4Gi \
  --cpu 4 \
  --max-instances 50 \
  --timeout 600
```

## Monitoring & Logging

### View Logs

```bash
# Stream logs in real-time
gcloud run logs read $SERVICE_NAME --region $REGION --follow

# View last 100 logs
gcloud run logs read $SERVICE_NAME --region $REGION --limit 100

# Filter for errors
gcloud run logs read $SERVICE_NAME --region $REGION --filter="severity=ERROR"

# Open in Cloud Console
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)' | xargs -I {} echo "https://console.cloud.google.com/run?project=$PROJECT_ID"
```

### Metrics to Monitor

- Request count and latency (p50, p95, p99)
- Error rate (5xx responses)
- Memory and CPU utilization
- Cold start latency

## Cost Optimization

```bash
# Cost-optimized deployment (development)
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 80 \
  --timeout 60
```

Cost tips:

1. Set `--min-instances 0` for dev (accept cold starts)
2. Use `--concurrency 80` to handle multiple requests per instance
3. Right-size resources based on actual metrics
4. Enable CPU throttling (default) to save costs when idle

## Troubleshooting

### Model not loading (model_loaded: false)

1. Check GCS credentials:

   ```bash
   gcloud auth application-default login
   gsutil ls gs://$GCS_BUCKET/models/
   ```

2. Verify environment variables are set:

   ```bash
   gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(spec.template.spec.containers[0].env)'
   ```

3. Check Cloud Run logs for download errors:

   ```bash
   gcloud run logs read $SERVICE_NAME --region $REGION --limit 50 | grep -i "gcs\|download\|error"
   ```

### Container failed to start

1. Check startup logs:

   ```bash
   gcloud run logs read $SERVICE_NAME --region $REGION --limit 100
   ```

2. Verify PORT is set to 8080:

   ```bash
   gcloud run services describe $SERVICE_NAME --region $REGION
   ```

3. Test image locally:

   ```bash
   docker run -p 8080:8080 ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1
   curl localhost:8080/health
   ```

### Out of memory errors

- Increase memory allocation:

  ```bash
  gcloud run deploy $SERVICE_NAME --memory 4Gi --region $REGION
  ```

### Authentication/permission errors

- Grant service account access to GCS:

  ```bash
  gsutil iam ch serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com:objectViewer gs://$GCS_BUCKET
  ```

## Security Best Practices

### Production Checklist

- [ ] Remove `--allow-unauthenticated` for private APIs
- [ ] Enable HTTPS (automatic with Cloud Run)
- [ ] Use Secret Manager for sensitive credentials
- [ ] Set up VPC connector for private database/resources
- [ ] Enable Cloud Armor for DDoS protection
- [ ] Implement rate limiting in API code
- [ ] Use least-privilege IAM roles for service accounts
- [ ] Enable audit logging

### Using Secret Manager

```bash
# Create secret for sensitive data
echo -n "gs://bucket/model.pt" | gcloud secrets create model-path --data-file=-

# Deploy with secret
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1 \
  --update-secrets MODEL_PATH=model-path:latest
```

## Cleanup

```bash
# Delete the Cloud Run service
gcloud run services delete $SERVICE_NAME --region $REGION

# Delete the Docker image from Artifact Registry
gcloud artifacts docker images delete ${REGION}-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$SERVICE_NAME:v1

# Delete GCS bucket and all data
gsutil rm -r gs://$GCS_BUCKET
```
