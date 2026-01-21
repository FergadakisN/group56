# M28: System Monitoring Implementation Guide

This guide documents the complete implementation of system monitoring for the Fish Species Classification API on Google Cloud Run, including Prometheus metrics, cloud monitoring, and alert systems.

## Overview

**M28 covers three key monitoring tasks:**

1. ✅ **Instrument API with system metrics** - Add Prometheus metrics for request tracking and latency
2. ✅ **Setup cloud monitoring** - Configure GCP Cloud Monitoring dashboard
3. ✅ **Create alert systems** - Set up automated alerts for anomalies

---

## Quick Start

### 1. Deploy to Cloud Run

```bash
# Build and push Docker image
docker build -f dockerfiles/api.dockerfile -t us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v4 .
docker push us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v4

# Deploy to Cloud Run
gcloud run deploy fish-classifier-api \
  --image us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v4 \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --update-env-vars MODEL_PATH=gs://fish-mlops-bucket/model.pt \
  --project fish-mlops
```

### 2. Verify Metrics Endpoint

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe fish-classifier-api \
  --region us-central1 \
  --format='value(status.url)' \
  --project fish-mlops)

# Send a test request
curl -X POST "$SERVICE_URL/predict" \
  -F file=@sample_image.jpg

# View metrics
curl "$SERVICE_URL/metrics" | grep fish_api
```

**Expected output:**

```txt
fish_api_requests_total{endpoint="/predict",method="POST"} 1.0
fish_api_prediction_latency_seconds_bucket{...} ...
```

### 3. Setup Alerts (Automated)

```bash
chmod +x scripts/setup_gcp_monitoring.sh
./scripts/setup_gcp_monitoring.sh fish-mlops fish-classifier-api us-central1 your-email@example.com
```

**Manual verification:**

- Check your email for verification link from GCP
- Click link to verify notification channel
- Wait 5 minutes for alert policies to activate

### 4. Verify Setup

```bash
# View notification channels
gcloud alpha monitoring channels list --project=fish-mlops

# View alert policies
gcloud alpha monitoring policies list --project=fish-mlops

# Send test traffic to verify alerts work
for i in {1..50}; do
  curl -X POST "$SERVICE_URL/predict" \
    -F file=@invalid.jpg -s -o /dev/null &
done
```

---

## Task 1: API Instrumentation with Prometheus

### Metrics Added

The Fish Classifier API now exposes four key Prometheus metrics on `/metrics`:

```yaml
fish_api_requests_total:
  Type: Counter
  Labels: [method, endpoint]
  Description: Total number of requests to the API
  Example: fish_api_requests_total{method="POST",endpoint="/predict"} 250

fish_api_errors_total:
  Type: Counter
  Labels: [method, endpoint, error_type]
  Description: Total number of errors by type
  Examples:
    - fish_api_errors_total{method="POST",endpoint="/predict",error_type="PredictionError"} 3
    - fish_api_errors_total{method="POST",endpoint="/predict",error_type="ModelNotLoaded"} 0
    - fish_api_errors_total{method="POST",endpoint="/predict",error_type="InvalidFileType"} 2

fish_api_prediction_latency_seconds:
  Type: Histogram
  Buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0]
  Description: Time taken for predictions in seconds
  Enables: p50, p95, p99 latency percentiles

fish_api_model_load_time_seconds:
  Type: Histogram
  Description: Model loading time on startup
  Used for: Startup performance monitoring
```

### Implementation Details

**Changes to `src/group_56/api.py`:**

```python
# Imports
from prometheus_client import Counter, Histogram, make_asgi_app

# Initialize metrics
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

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())
```

**Metric collection in endpoints:**

- `/predict` endpoint tracks:
  - Request count (incremented at start)
  - Error type and count (if validation fails)
  - Prediction latency (recorded at end)
  - Model load time (measured during startup)

### Testing Metrics Locally

```bash
# 1. Install dependencies
pip install prometheus-client

# 2. Start the API
python -m uvicorn src.group_56.api:app --host 0.0.0.0 --port 8080

# 3. In another terminal, send test requests
curl -X POST http://localhost:8080/predict \
  -F file=@test_image.jpg

# 4. View metrics
curl http://localhost:8080/metrics | grep fish_api

# Expected output:
# fish_api_requests_total{endpoint="/predict",method="POST"} 1.0
# fish_api_prediction_latency_seconds_bucket{le="0.05",method="POST",...} 0.0
# fish_api_prediction_latency_seconds_bucket{le="0.1",method="POST",...} 1.0
```

---

## Task 2: Setup Cloud Monitoring

### Cloud Run Built-in Metrics

Google Cloud Run automatically collects:

- **Request Count**: Number of requests per minute
- **Request Latencies**: p50, p95, p99 latencies
- **Error Rate**: Percentage of 5xx responses
- **Memory Usage**: Per-instance memory consumption
- **CPU Utilization**: Per-instance CPU percentage
- **Instance Count**: Number of active instances

### Creating a Monitoring Dashboard

#### Option A: Manual Setup via Cloud Console

1. Go to **Cloud Console** → **Monitoring** → **Dashboards**
2. Click **Create Dashboard**
3. Add charts:
   - Request Rate: `run.googleapis.com/request_count`
   - Error Rate (5xx): Filter by `metric.response_code_class="5xx"`
   - Latency (P95): `run.googleapis.com/request_latencies` with percentile=95
   - Memory Usage: `run.googleapis.com/container_memory_utilization`
   - CPU Utilization: `run.googleapis.com/container_cpu_utilization`

#### Option B: Using gcloud CLI

```bash
# List available metrics for your service
gcloud monitoring metrics-descriptors list \
  --filter='metric.type:run.googleapis.com' \
  --project=fish-mlops

# View recent metrics
gcloud monitoring time-series list \
  --filter='resource.service_name="fish-classifier-api"' \
  --project=fish-mlops
```

### Key Metrics to Monitor

| Metric | Target | Alert if | Priority |
| -------- | -------- | ---------- | ---------- |
| Error Rate (5xx) | <1% | >5% | 🔴 High |
| P95 Latency | <1000ms | >2000ms | 🟡 Medium |
| Request Count | >10 req/min | <1 req/5min | 🟡 Medium |
| Memory Usage | <70% | >90% | 🟡 Medium |
| Cold Starts | <5s | >10s | 🟢 Low |

### Viewing Metrics in Cloud Console

1. Go to **Cloud Console** → **Cloud Run** → **fish-classifier-api**
2. Select **Metrics** tab
3. Choose time range and metrics from dropdown
4. Filter by revision (optional)

```txt
Key data points to check:
- Requests per second
- Average response time
- Error percentage
- Instance scaling pattern
```

---

## Task 3: GCP Alert Systems

### Setup Email Notification Channel

**Via Cloud Console:**

1. Go to **Monitoring** → **Alerting** → **Notification Channels**
2. Click **Create Channel**
3. Select **Email**
4. Enter your email: `your-email@example.com`
5. Click **Create Channel**
6. **Verify** the email by clicking the link in your inbox

**Via gcloud CLI:**

```bash
gcloud alpha monitoring channels create \
  --display-name="fish-classifier-notifications" \
  --type=email \
  --channel-labels=email_address="your-email@example.com" \
  --project=fish-mlops
```

### Create Alert Policies

#### Alert Policy 1: High Error Rate

**Condition:**

- Metric: `cloud.run/request_count` with `response_code_class="5xx"`
- Service: `fish-classifier-api`
- Threshold: >5% error rate
- Duration: 5 minutes

**Steps:**

1. Go to **Monitoring** → **Alerting** → **Policies** → **Create Policy**
2. Add Condition:
   - Resource Type: `cloud_run_revision`
   - Metric: `run.googleapis.com/request_count`
   - Filter: `resource.service_name="fish-classifier-api"` AND `metric.response_code_class="5xx"`
   - Condition: Ratio to total requests > 0.05
   - Duration: 300s
3. Add Notification Channel (email)
4. Save Policy

#### Alert Policy 2: High Latency

**Condition:**

- Metric: `run.googleapis.com/request_latencies` (P95 percentile)
- Service: `fish-classifier-api`
- Threshold: >2000ms (2 seconds)
- Duration: 5 minutes

#### Alert Policy 3: No Requests (Service Unresponsive)

**Condition:**

- Metric: `run.googleapis.com/request_count`
- Service: `fish-classifier-api`
- Threshold: 0 requests for 10 minutes
- Duration: 600s

### Automated Setup Script

Use the provided setup script:

```bash
chmod +x scripts/setup_gcp_monitoring.sh

# Run with defaults
./scripts/setup_gcp_monitoring.sh

# Or with custom parameters
./scripts/setup_gcp_monitoring.sh fish-mlops fish-classifier-api us-central1 your-email@example.com
```

### Testing Alerts

**Trigger Error Alert:**

```bash
# Send multiple invalid requests to trigger error rate alert
SERVICE_URL="https://fish-classifier-api-*.us-central1.run.app"

for i in {1..50}; do
  curl -X POST "$SERVICE_URL/predict" \
    -F file=@/invalid/path.jpg \
    -s -o /dev/null &
done

wait
echo "Sent 50 error requests - monitor your email for alert"
```

**Check Alert Status:**

```bash
# List all alert policies
gcloud alpha monitoring policies list --project=fish-mlops

# View alert history
gcloud alpha monitoring incidents list --project=fish-mlops

# View notification channels
gcloud alpha monitoring channels list --project=fish-mlops
```

---

## Monitoring Dashboard

### Recommended Dashboard Layout

```txt
┌─────────────────────────────────────────────┐
│     Fish Classifier API - Monitoring         │
├─────────────────────────────────────────────┤
│                                              │
│  [Requests/min]      [Error Rate %]         │
│  ▲                    ▲                      │
│  │ 250/min           │ 2% (Alert: 5%)       │
│  └──────────         └──────────            │
│                                              │
│  [P95 Latency (ms)]   [Active Instances]    │
│  ▲                    ▲                      │
│  │ 450ms             │ 2 instances          │
│  │ (Alert: 2000ms)   │                      │
│  └──────────         └──────────            │
│                                              │
│  [Memory Usage %]     [CPU Usage %]         │
│  ▲                    ▲                      │
│  │ 65%               │ 45%                  │
│  └──────────         └──────────            │
└─────────────────────────────────────────────┘
```

### Key Monitoring Commands

```bash
# View current metrics (last 10 minutes)
gcloud monitoring time-series list \
  --filter='resource.service_name="fish-classifier-api"' \
  --project=fish-mlops \
  --format=table

# Stream logs in real-time
gcloud run logs read fish-classifier-api \
  --region=us-central1 \
  --follow \
  --project=fish-mlops

# Get service status
gcloud run services describe fish-classifier-api \
  --region=us-central1 \
  --project=fish-mlops

# View all errors in last hour
gcloud logging read 'resource.service_name="fish-classifier-api" AND severity>=ERROR' \
  --limit 100 \
  --project=fish-mlops
```

---

## M28 Completion Checklist

- [x] Added Prometheus metrics to FastAPI API
  - [x] Counter for request count (by method, endpoint)
  - [x] Counter for error count (by type)
  - [x] Histogram for prediction latency
  - [x] Histogram for model load time
  
- [x] Exposed `/metrics` endpoint for Prometheus scraping

- [x] Setup GCP Cloud Monitoring
  - [x] Enable Monitoring API
  - [x] Create notification channel (email)
  - [x] Configure alert policies (3 recommended)
  - [x] Set up monitoring dashboard

- [x] Created alert systems
  - [x] High error rate alert (>5% errors)
  - [x] High latency alert (P95 > 2s)
  - [x] Service unresponsive alert (no requests for 10 min)

- [x] Testing & Documentation
  - [x] Verified metrics endpoint locally
  - [x] Tested alert triggering
  - [x] Documented monitoring best practices

---

## Troubleshooting

### Metrics not showing

1. Ensure `prometheus-client` is installed: `pip install prometheus-client`
2. Restart the Cloud Run service after code changes
3. Send a request to generate metrics: `curl https://<your-service-url>/health`
4. Check `/metrics` endpoint: `curl https://<your-service-url>/metrics`

### Alerts not firing

1. Verify email channel is verified in Cloud Console
2. Check alert policy conditions are correct
3. Generate test traffic to trigger alert condition
4. Check alert history: `gcloud alpha monitoring incidents list`

### High latency detected

1. Check model loading time: `grep "model_load_time" /metrics`
2. Verify instance has enough CPU/memory
3. Scale up: `gcloud run deploy fish-classifier-api --memory 4Gi --cpu 4`

---

## References

- [Prometheus Client Python Docs](https://github.com/prometheus/client_python)
- [GCP Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [Cloud Run Metrics](https://cloud.google.com/run/docs/monitoring)
- [Alert Policy Best Practices](https://cloud.google.com/monitoring/alert-policies)
